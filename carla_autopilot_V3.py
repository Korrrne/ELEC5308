#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于用户多摄像头+LiDAR代码的【自动驾驶骨架 · v3（感知升级）】
- Lane：HLS 颜色阈值 + Sobel 梯度 → 逆透视（IPM）→ 滑窗二次曲线拟合 → 中线像素偏差
- LiDAR：前向窗口 + 去地近似（z带限制）+ 网格聚类 → 最近簇距离，稳健于噪点
- 控制：横向/纵向 PID（与 v1 相同），但横向误差经 EMA 平滑
适配：CARLA 0.9.15 + Python 3.7（Windows）
"""

import sys, os, glob, time, math, random
from collections import deque
import numpy as np
import cv2

# ==== 配置区：按需修改 ====
CARLA_ROOT = r"D:/Carla/WindowsNoEditor"  # 你的 CARLA 安装目录
HOST = 'localhost'
PORT = 2000
SYNC_DT = 0.05  # 20Hz
IMG_W, IMG_H = 640, 480
FOV_FRONT = 90
FOV_SIDE  = 110
LIDAR_RANGE = 30.0

# ==== 先确保能 import carla ====
egg_glob = os.path.join(
    CARLA_ROOT, "PythonAPI", "carla", "dist",
    f"carla-*{sys.version_info.major}.{sys.version_info.minor}-win-amd64.egg"
)
eggs = glob.glob(egg_glob)
if eggs and eggs[0] not in sys.path:
    sys.path.append(eggs[0])

try:
    import carla
except Exception as e:
    raise RuntimeError("无法导入 carla 模块。请确认：\n"
                       "1) 已安装与当前 Python 小版本匹配的 CARLA egg 或 pip wheel；\n"
                       "2) CARLA_ROOT 路径正确；\n"
                       "3) 正在使用 Python 3.7 运行（对于 0.9.15）。") from e

# ============== 工具函数 ==============

def to_bgr_image(image: 'carla.Image') -> np.ndarray:
    """CARLA 图像 → BGR uint8 (H, W, 3)"""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]  # 先 RGBA，取 RGB
    return bgr

def lidar_to_xyz(lidar: 'carla.LidarMeasurement') -> np.ndarray:
    """LiDAR → (N,3) 点数组（传感器坐标：x前 y右 z上）"""
    data = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
    return data[:, :3]

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

class PID:
    def __init__(self, kp, ki, kd, lim_min=-1.0, lim_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lim_min, self.lim_max = lim_min, lim_max
        self.ei = 0.0
        self.last_e = 0.0

    def step(self, e, dt):
        self.ei += e*dt
        de = (e - self.last_e)/dt if dt > 0 else 0.0
        u = self.kp*e + self.ki*self.ei + self.kd*de
        self.last_e = e
        return clamp(u, self.lim_min, self.lim_max)

# ============== 主类 ==============

class SensorBus:
    """管理多摄像头 + LiDAR 的共享数据"""
    def __init__(self):
        self.cams = {'front': None, 'left': None, 'right': None, 'rear': None}
        self.lidar_points = None   # (N,3) for 控制
        self.lidar_view   = None   # 可视化图像 for 展示

class CarlaAutopilotFromSensors:
    def __init__(self, host=HOST, port=PORT):
        self.client = carla.Client(host, port); self.client.set_timeout(10.0)
        self.world  = self.client.get_world()
        self.map    = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        self.vehicle = None
        self.cams    = {k: None for k in ['front','left','right','rear']}
        self.lidar   = None
        self.bus     = SensorBus()

        self.spectator = self.world.get_spectator()
        self.waypoints = []  # 仅用于可视化
        self._orig_settings = self.world.get_settings()

        # 控制器与参数
        self.dt = SYNC_DT
        self.target_speed = 8.0     # m/s ≈ 28.8 km/h
        self.caution_d    = 18.0    # 进入减速区
        self.safe_stop_d  = 5.0     # 必须刹停
        self.steer_pid = PID(kp=0.8, ki=0.0, kd=0.2, lim_min=-0.8, lim_max=0.8)
        self.speed_pid = PID(kp=0.25,  ki=0.05, kd=0.02,  lim_min=0.0, lim_max=1.0)

        # —— 感知状态（新增） ——
        self.lane_state = {
            'fit_l': None, 'fit_r': None,     # 上一帧左/右车道二次曲线系数
            'xc_ema': None                    # 中线位置的指数滑动平均
        }
        self.ema_alpha = 0.2
        self.persp_cache = {}  # (w,h) -> (M, Minv)

    # ---------- 连接/生成 ----------
    def enable_sync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

    def disable_sync(self):
        self.world.apply_settings(self._orig_settings)

    def spawn_vehicle(self):
        bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        spawn = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.try_spawn_actor(bp, spawn)
        if not self.vehicle:
            raise RuntimeError("车辆生成失败，可能是生成点被占用。")
        self.vehicle.set_autopilot(False)  # 关键：自管
        print(f"[OK] 车辆已生成 @ {spawn.location}")

    def attach_cameras(self):
        cfgs = {
            'front': dict(tf=carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0, yaw=0)),   fov=FOV_FRONT),
            'left':  dict(tf=carla.Transform(carla.Location(x=0.0, y=-1.5, z=2.2), carla.Rotation(pitch=0, yaw=-90)), fov=FOV_SIDE),
            'right': dict(tf=carla.Transform(carla.Location(x=0.0, y= 1.5, z=2.2), carla.Rotation(pitch=0, yaw= 90)), fov=FOV_SIDE),
            'rear':  dict(tf=carla.Transform(carla.Location(x=-2.0, z=2.4), carla.Rotation(pitch=0, yaw=180)), fov=FOV_FRONT),
        }
        for name, cfg in cfgs.items():
            bp = self.bp_lib.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(IMG_W))
            bp.set_attribute('image_size_y', str(IMG_H))
            bp.set_attribute('fov', str(cfg['fov']))
            cam = self.world.spawn_actor(bp, cfg['tf'], attach_to=self.vehicle)
            cam.listen(self._make_cam_cb(name))
            self.cams[name] = cam
        print("[OK] 多摄像头就绪：", ", ".join(self.cams.keys()))

    def _make_cam_cb(self, name):
        def _cb(image):
            self.bus.cams[name] = to_bgr_image(image)
        return _cb

    def attach_lidar(self):
        bp = self.bp_lib.find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '180000')
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('range', str(LIDAR_RANGE))
        bp.set_attribute('upper_fov', '10')
        bp.set_attribute('lower_fov', '-25')
        tf = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.lidar = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        self.lidar.listen(self._lidar_cb)
        print("[OK] LiDAR 就绪")

    def _lidar_cb(self, lidar):
        pts = lidar_to_xyz(lidar)
        self.bus.lidar_points = pts
        self.bus.lidar_view = self._lidar_birdview(pts)

    def _lidar_birdview(self, pts: np.ndarray, size=420) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        if pts is None or len(pts) == 0:
            return img
        R = min(LIDAR_RANGE, 40.0)
        m = np.linalg.norm(pts[:, :2], axis=1) < R
        pts = pts[m]
        if pts.size == 0:
            return img
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        s = size / (2*R)
        xi = (size//2 + (y * s)).astype(int)
        yi = (size//2 - (x * s)).astype(int)
        m = (xi>=0)&(xi<size)&(yi>=0)&(yi<size)
        xi, yi, z = xi[m], yi[m], z[m]
        z_norm = np.clip((z + 2.0)/4.0, 0, 1)
        for xx, yy, zz in zip(xi, yi, z_norm):
            img[yy, xx] = (int((1-zz)*255), int(zz*255), 0)
        cv2.circle(img, (size//2, size//2), 4, (255,255,255), -1)
        cv2.putText(img, "FRONT", (size//2-20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return img

    # ---------- 透视变换（新增） ----------
    def _get_persp_mats(self, w, h):
        """缓存/生成 逆透视矩阵 M 与逆变换 Minv（经验点位，适用于车道视角；可按相机重标定优化）"""
        key = (w, h)
        if key in self.persp_cache:
            return self.persp_cache[key]
        # 源点（原图）——近似包住车道区域的四边形
        src = np.float32([
            [w*0.12, h*0.95],
            [w*0.88, h*0.95],
            [w*0.58, h*0.65],
            [w*0.42, h*0.65]])
        # 目标点（鸟瞰图）——映射为直角矩形
        dst = np.float32([
            [w*0.20, h*0.98],
            [w*0.80, h*0.98],
            [w*0.80, h*0.10],
            [w*0.20, h*0.10]])
        M    = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        self.persp_cache[key] = (M, Minv)
        return M, Minv

    # ---------- 感知：车道（升级） ----------
    def detect_lane_error_px(self, bgr_front: np.ndarray) -> (float, int):
        """
        估计“车道中心相对图像中心”的像素偏差（右为正）。
        升级点：HLS 颜色阈值 + Sobel 梯度 → IPM → 滑窗拟合 → 逆映射回原图求误差，并做 EMA 平滑。
        返回：(err_px, w) —— 仍以原图宽度归一化，兼容旧控制器。
        """
        if bgr_front is None:
            return 0.0, IMG_W

        h, w = bgr_front.shape[:2]
        M, Minv = self._get_persp_mats(w, h)

        # 1) 下半幅 ROI
        roi = bgr_front[int(h*0.45):, :]
        # 2) 颜色阈值（白/黄线）
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        H, L, S = cv2.split(hls)
        # 白线：高亮度
        white = (L > 200) & (S > 60)
        # 黄线：H 在 [15, 35] 度附近（CARLA 黄线偏暖），S/L 要足够
        yellow = (H > 15) & (H < 35) & (S > 70) & (L > 120)
        mask_color = (white | yellow).astype(np.uint8) * 255

        # 3) 梯度增强
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
        mask_grad = (mag > 80).astype(np.uint8) * 255

        # 4) 融合 + 形态学
        mask = cv2.bitwise_or(mask_color, mask_grad)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # 5) 逆透视到鸟瞰
        # 注意：roi 高度 < 原图高度，需构造全尺寸空白再 warp，以保持矩阵对应
        full = np.zeros((h, w), dtype=np.uint8)
        full[int(h*0.45):, :] = mask
        bev = cv2.warpPerspective(full, M, (w, h), flags=cv2.INTER_LINEAR)

        # 6) 直方图基线 & 滑窗聚点
        histogram = np.sum(bev[int(h*0.55):, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = h // nwindows
        nonzero = bev.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 60
        minpix = 50

        leftx_current  = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for win in range(nwindows):
            win_y_low  = h - (win + 1) * window_height
            win_y_high = h - win * window_height
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds  = np.concatenate(left_lane_inds)  if left_lane_inds else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        # 7) 二次曲线拟合（若某侧像素太少，使用上一帧拟合）
        fit_l, fit_r = None, None
        if left_lane_inds.size > 200:
            fit_l = np.polyfit(nonzeroy[left_lane_inds],  nonzerox[left_lane_inds],  2)
            self.lane_state['fit_l'] = fit_l
        elif self.lane_state['fit_l'] is not None:
            fit_l = self.lane_state['fit_l']

        if right_lane_inds.size > 200:
            fit_r = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            self.lane_state['fit_r'] = fit_r
        elif self.lane_state['fit_r'] is not None:
            fit_r = self.lane_state['fit_r']

        # 如果仍不可用，兜底：用图像中线
        if fit_l is None or fit_r is None:
            xc_bev = w // 2
        else:
            y_eval = h - 1
            xl = fit_l[0]*y_eval**2 + fit_l[1]*y_eval + fit_l[2]
            xr = fit_r[0]*y_eval**2 + fit_r[1]*y_eval + fit_r[2]
            xc_bev = int((xl + xr) * 0.5)

        # 8) 将鸟瞰底部中线点逆映射回原图，求像素误差
        pt_bev = np.array([[[float(xc_bev), float(h - 1)]]], dtype=np.float32)  # shape (1,1,2)
        pt_img = cv2.perspectiveTransform(pt_bev, Minv).reshape(2)
        x_center_img = float(pt_img[0])  # 原图 x
        err_px = x_center_img - (w / 2.0)

        # 9) EMA 平滑
        if self.lane_state['xc_ema'] is None:
            self.lane_state['xc_ema'] = x_center_img
        else:
            a = self.ema_alpha
            self.lane_state['xc_ema'] = (1 - a) * self.lane_state['xc_ema'] + a * x_center_img
        err_px = self.lane_state['xc_ema'] - (w / 2.0)

        return float(err_px), w

    # ---------- 感知：前向最近障碍（升级：网格聚类） ----------
    def closest_ahead_distance(self, pts: np.ndarray,
                               x_max=28.0, y_half=1.8, z_min=-1.8, z_max=0.6,
                               grid=0.8, min_points=12) -> float:
        """
        以网格聚类替代“全体点最小距离”，降低噪点误判；不依赖 sklearn。
        - 仅保留前方 x∈(0, x_max)，|y|<y_half，z∈[z_min, z_max] 的点（粗略去地）
        - 量化到 (grid × grid) 网格作为“簇”，筛选点数>=min_points 的簇
        - 返回最近簇的极小径向距离；如无簇，则退回所有点的最小径向距离；再无则 +inf
        """
        if pts is None or len(pts) == 0:
            return float('inf')

        x = pts[:,0]; y = pts[:,1]; z = pts[:,2]
        m = (x > 0.0) & (x < x_max) & (np.abs(y) < y_half) & (z > z_min) & (z < z_max)
        if not np.any(m):
            return float('inf')
        xf, yf = x[m], y[m]
        df = np.sqrt(xf*xf + yf*yf)

        # 网格聚类
        gx = np.floor(xf / grid).astype(np.int32)
        gy = np.floor(yf / grid).astype(np.int32)
        keys, inverse, counts = np.unique(np.stack([gx, gy], axis=1), axis=0, return_inverse=True, return_counts=True)

        # 筛出点数达标的簇，计算每簇最小距离
        best = float('inf')
        for k_idx, cnt in enumerate(counts):
            if cnt < min_points:
                continue
            # 属于该簇的点索引
            idx = (inverse == k_idx)
            dmin = float(df[idx].min())
            if dmin < best:
                best = dmin

        if best < float('inf'):
            return best
        # 退回所有点的最小距离（避免偶尔阈值过严导致 0 簇）
        return float(df.min()) if df.size else float('inf')

    # ---------- 控制 ----------
    def compute_control(self):
        # 速度（m/s）
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) if v is not None else 0.0

        # --------- 默认量，避免未赋值 ---------
        e_lat = 0.0
        d_front = float('inf')

        # ===== LIDAR 最近前方距离 =====
        pts = getattr(self.bus, 'lidar_points', None)
        if pts is not None:
            try:
                d_front = float(self.closest_ahead_distance(pts))
            except Exception:
                d_front = float('inf')

        # ===== 相机：若还没帧，先直行蠕动，避免报错 =====
        front = self.bus.cams.get('front') if hasattr(self.bus, 'cams') else None
        if front is None:
            throttle = 0.28 if speed < 3.0/3.6 else 0.0   # 给点油让车慢慢动起来
            steer_cmd = 0.0
            brake = 0.0
            return throttle, steer_cmd, brake, speed, d_front, e_lat

        # ===== 车道横向误差归一化到 [-1,1] =====
        err_px, w = self.detect_lane_error_px(front)
        if w and w > 0:
            e_lat = float(err_px) / float(w/2.0)
            # 保护一下，别让异常值炸开
            e_lat = max(-1.5, min(1.5, e_lat))
        else:
            e_lat = 0.0

        # ===== 转向 PID（PD 即可；Ki=0）=====
        pid_out = self.steer_pid.step(e_lat, self.dt)
        if pid_out is None or (isinstance(pid_out, float) and math.isnan(pid_out)):
            pid_out = 0.0

        # 最终转向（限幅）
        steer_cmd = float(np.clip(pid_out, -0.6, 0.6))

        # ===== 动态目标车速（m/s）=====
        dyn_target = float(self.target_speed)
        if math.isfinite(d_front) and d_front < self.caution_d:
            num = max(0.0, d_front - self.safe_stop_d)
            den = max(1e-3, self.caution_d - self.safe_stop_d)
            dyn_target = self.target_speed * (num/den)

        # 速度 PID：目标 - 当前
        v_err = dyn_target - speed
        throttle = float(self.speed_pid.step(v_err, self.dt) or 0.0)
        brake = 0.0

        # 近距策略：允许低速蠕行验证转向（不要一上来就锁死）
        if math.isfinite(d_front) and d_front < self.safe_stop_d:
            if speed < 1.0:                 # <1 m/s 给轻油门
                throttle = max(throttle, 0.28)
                brake = 0.0
            else:
                throttle = 0.0
                brake = max(brake, 0.35)

        # 夹紧
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake    = float(np.clip(brake,    0.0, 1.0))
        return throttle, steer_cmd, brake, speed, d_front, e_lat

    # ---------- 展示/观众机 ----------
    def update_spectator(self):
        try:
            vt = self.vehicle.get_transform()
            f  = vt.get_forward_vector()
            cam_loc = vt.location - 8.0 * f + carla.Location(z=3.0)
            cam_rot = carla.Rotation(pitch=-12.0, yaw=vt.rotation.yaw)
            self.spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        except Exception:
            pass

    def draw_waypoints(self, n=12, step=18.0):
        self.waypoints.clear()
        wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
        for i in range(n):
            self.waypoints.append(wp)
            nxt = wp.next(step)
            if nxt: wp = nxt[0]
            else: break
        for i in range(len(self.waypoints)-1):
            a = self.waypoints[i].transform.location
            b = self.waypoints[i+1].transform.location
            self.world.debug.draw_arrow(a, b, thickness=0.1, arrow_size=0.2,
                                        color=carla.Color(0,255,0), life_time=0.2, persistent_lines=False)

    # ---------- 主流程 ----------
    def run(self, show_windows=True):
        print("[*] 启动同步模式...")
        self.enable_sync()

        print("[*] 生成车辆与传感器...")
        self.spawn_vehicle()
        self.attach_cameras()
        self.attach_lidar()

        for _ in range(5):
            self.world.tick()
            time.sleep(0.01)

        if show_windows:
            cv2.namedWindow("Front",  cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Left",   cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Right",  cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Rear",   cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("LiDAR",  cv2.WINDOW_AUTOSIZE)

        print("[*] 进入控制循环：按 q 退出")
        try:
            t0 = time.time()
            while True:
                self.world.tick()
                self.update_spectator()
                self.draw_waypoints()

                # 计算控制量
                throttle, steer, brake, speed, d_front, e_lat = self.compute_control()

                # 安全限幅（可视化前先限幅）
                steer   = float(np.clip(steer, -0.9, 0.9))
                throttle = float(np.clip(throttle, 0.0, 1.0))
                brake    = float(np.clip(brake, 0.0, 1.0))

                # 以车辆实际速度为准（防止 compute_control 返回的 speed 为空/过期）
                v = self.vehicle.get_velocity()
                speed_ms = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) if v is not None else 0.0

                # 统一打印：看得到“算出来的转向值”有没有变化
                print(f"[CTRL] steer={steer:+.3f}  throttle={throttle:.2f}  brake={brake:.2f}  "
                    f"speed={speed_ms*3.6:4.1f} km/h  e_lat={e_lat:+.3f}  d_front={d_front:4.1f} m")

                # 真正下发控制（只下发一次，避免被覆盖）
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=throttle, steer=steer, brake=brake, manual_gear_shift=False
                ))

                # --------- 可视化 ---------
                if show_windows:
                    for name in ["Front","Left","Right","Rear"]:
                        key = name.lower()
                        img = self.bus.cams.get(key, None)
                        if img is None:
                            # 若没有 IMG_H/IMG_W 常量，可用相机实际分辨率或 img 的 shape
                            blank = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(blank, f"Waiting {name}...", (30, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                            cv2.imshow(name, blank)
                        else:
                            vis = img.copy()
                            cv2.putText(vis,
                                f"{name} | v={speed_ms*3.6:4.1f} km/h | d_front={d_front:4.1f} m | e_lat={e_lat:+.3f}",
                                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            if name=="Front":
                                h, w = vis.shape[:2]
                                cv2.line(vis, (w//2, h-60), (w//2, h), (0,255,255), 2)  # 车辆中心参考线
                            cv2.imshow(name, vis)

                    if self.bus.lidar_view is not None:
                        cv2.imshow("LiDAR", self.bus.lidar_view)

                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
        finally:
            self.cleanup()

    def cleanup(self):
        print("[*] 清理资源...")
        try:
            if self.cams:
                for _, cam in self.cams.items():
                    if cam is not None:
                        cam.stop(); cam.destroy()
            if self.lidar is not None:
                self.lidar.stop(); self.lidar.destroy()
            if self.vehicle is not None:
                self.vehicle.destroy()
            cv2.destroyAllWindows()
        except Exception as e:
            print("清理时出错：", e)
        finally:
            self.disable_sync()
            print("[OK] 结束")

# 入口
def main():
    print("=== 自主自动驾驶骨架（相机+LiDAR）· v2（感知升级） ===")
    print("请确保 CarlaUE4/CarlaUE5 已启动，地图加载完成。")
    app = CarlaAutopilotFromSensors(HOST, PORT)
    app.run(show_windows=True)

if __name__ == "__main__":
    main()
