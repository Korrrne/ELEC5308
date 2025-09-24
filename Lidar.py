# t1_bev_vehicle_realtime_slow_fix.py — BEV 实时帧 + 限速 ~8 km/h + 平滑恢复（不改终点）

import glob, os, sys, math, random, time
import numpy as np, cv2

# ===== paths (按需修改) =====
carla_root = r"D:\CARLA9\WindowsNoEditor\PythonAPI"
sys.path.append(carla_root)
sys.path.append(os.path.join(carla_root, "carla"))
sys.path.append(os.path.join(carla_root, "agents"))
try:
    import sys as _sys
    sys.path.append(glob.glob(
        os.path.join(carla_root, "dist", "carla-*%d.%d-%s.egg" % (
            _sys.version_info.major, _sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
    )[0])
except Exception:
    pass

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

# ===== 参数 =====
LIDAR_RANGE_M = 25
SYNC_MODE = True
_headless = False

# BEV 实时帧
BEV_W, BEV_H = 1000, 1000
BEV_PX_PER_M = 18
_bev_img = np.zeros((BEV_H, BEV_W), np.float32)

# 限速
DESIRED_SPEED_MPS = 2.3   # ≈ 8.3 km/h

# 恢复参数（不改终点）
RECOVER_COOLDOWN_S = 1.2  # 冷却期：只刹车+回正方向
RECOVER_BRAKE = 0.45
RECOVER_STEER_CAP = 0.25  # 冷却期方向限幅
RECOVER_TRIGGER_STEER = 0.75
RECOVER_TRIGGER_SPEED = 0.6  # m/s
RECOVER_TRIGGER_TICKS = 10   # 满足条件累计 ~1s 才恢复（tick=0.1s 时）

# ===== 状态 =====
_lidar = None
_vehicle_ref = None
_last_plan_time = 0.0
REPLAN_COOLDOWN_S = 2.0
_recover_until = 0.0
_stuck_counter = 0

# =========================================
# LiDAR 回调：车辆坐标系 BEV 实时帧（无拖影）
# =========================================
def cb_lidar(meas: carla.LidarMeasurement, cfg=None):
    global _bev_img
    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
    xy = pts[:, :2]
    cx, cy = BEV_W // 2, BEV_H // 2
    px = (cx + xy[:, 1] * BEV_PX_PER_M).astype(np.int32)
    py = (cy - xy[:, 0] * BEV_PX_PER_M).astype(np.int32)
    mask = (px >= 0) & (px < BEV_W) & (py >= 0) & (py < BEV_H)
    _bev_img = np.zeros((BEV_H, BEV_W), np.float32)
    np.add.at(_bev_img, (py[mask], px[mask]), 1.0)
    if not _headless:
        m = float(_bev_img.max())
        bev_u8 = (_bev_img / m * 255).astype(np.uint8) if m > 0 else _bev_img.astype(np.uint8)
        bev_color = cv2.applyColorMap(bev_u8, cv2.COLORMAP_TURBO)
        cv2.imshow('BEV Instant (Vehicle-Centric)', bev_color); cv2.waitKey(1)

# ---- 车道中心+航向对正 ----
def snap_to_lane_center(vehicle, world):
    wp = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if wp is None: return
    tf = wp.transform
    tf.location.z = vehicle.get_location().z
    vehicle.set_transform(tf)

def pick_far_destination(world, origin_loc, min_dist=140.0):
    spawns = world.get_map().get_spawn_points(); random.shuffle(spawns)
    for sp in spawns:
        if sp.location.distance(origin_loc) >= min_dist:
            return sp.location
    return max(spawns, key=lambda s: s.location.distance(origin_loc)).location

def set_agent_destination(agent: BehaviorAgent, world, start_loc=None):
    global _last_plan_time
    if start_loc is None:
        start_loc = agent._vehicle.get_location()
    dest = pick_far_destination(world, start_loc)
    agent.set_destination(dest)   # 新版 API：只传终点
    _last_plan_time = time.time()
    print("[Agent] New destination set. Dist = %.1f m" % start_loc.distance(dest))

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def main():
    global _lidar, _vehicle_ref, _last_plan_time, _recover_until, _stuck_counter

    client = carla.Client('127.0.0.1', 2000); client.set_timeout(60.0)
    world = client.get_world(); tm = client.get_trafficmanager()

    old_settings = world.get_settings()
    if SYNC_MODE:
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 0.1  # 10 Hz
        world.apply_settings(new_settings)
        tm.set_synchronous_mode(True)

    actors = []
    try:
        bp_lib = world.get_blueprint_library()

        # 车辆
        v_bp = bp_lib.filter("vehicle.*model3*")[0]
        spawns = world.get_map().get_spawn_points(); random.shuffle(spawns)
        vehicle = None
        for sp in spawns:
            v = world.try_spawn_actor(v_bp, sp)
            if v is not None: vehicle = v; break
        if vehicle is None:
            vehicles = world.get_actors().filter('vehicle.*')
            if len(vehicles) == 0: raise RuntimeError("No vehicle available.")
            print("[WARN] Attach to existing vehicle."); vehicle = vehicles[0]
        else:
            actors.append(vehicle)

        _vehicle_ref = vehicle
        vehicle.set_autopilot(False)
        snap_to_lane_center(vehicle, world)

        # LiDAR（降负载）
        lbp = bp_lib.find('sensor.lidar.ray_cast')
        lbp.set_attribute('range', str(LIDAR_RANGE_M))
        lbp.set_attribute('rotation_frequency', '15')
        lbp.set_attribute('channels', '36')
        lbp.set_attribute('points_per_second', '200000')
        lbp.set_attribute('upper_fov', '5.0')
        lbp.set_attribute('lower_fov', '-30.0')
        lbp.set_attribute('sensor_tick', '0.2')
        _lidar = world.spawn_actor(lbp, carla.Transform(carla.Location(x=0, z=1.8)), attach_to=vehicle)
        actors.append(_lidar); _lidar.listen(cb_lidar)

        # Agent
        agent = BehaviorAgent(vehicle, behavior='normal')
        agent._local_planner.set_speed(DESIRED_SPEED_MPS)
        set_agent_destination(agent, world, vehicle.get_location())

        start_time = time.time(); log_t = 0.0

        while True:
            if SYNC_MODE: world.tick()
            else: world.wait_for_tick()

            now = time.time()

            # 仅在冷却窗口外检查到达
            if (now - _last_plan_time) >= REPLAN_COOLDOWN_S and agent.done():
                set_agent_destination(agent, world, vehicle.get_location())

            control = agent.run_step()

            # 转向限幅与平滑（基础）
            elapsed = now - start_time
            steer = float(control.steer)
            if elapsed < 2.0:
                steer = np.clip(steer, -0.35, 0.35)
            else:
                last_steer = float(getattr(main, "_last_steer", 0.0))
                steer = 0.7 * last_steer + 0.3 * steer
                steer = float(np.clip(steer, -0.9, 0.9))
            main._last_steer = steer

            speed = get_speed(vehicle)  # m/s

            # ====== 恢复阶段：只刹车、方向回正、限幅；不改终点 ======
            if now < _recover_until:
                control.throttle = 0.0
                control.brake = max(control.brake, RECOVER_BRAKE)
                steer = float(np.clip(steer, -RECOVER_STEER_CAP, RECOVER_STEER_CAP))
                control.steer = steer
                control.manual_gear_shift = False
                vehicle.apply_control(control)
            else:
                # 非恢复阶段：正常限速器
                control.steer = steer
                if speed > DESIRED_SPEED_MPS + 0.4:
                    control.throttle = 0.0
                    control.brake = max(control.brake, 0.3)
                elif speed < DESIRED_SPEED_MPS - 0.4:
                    control.brake = 0.0
                    control.throttle = min(control.throttle, 0.35)
                else:
                    control.throttle = min(control.throttle, 0.25)
                    control.brake = min(control.brake, 0.2)
                control.manual_gear_shift = False
                vehicle.apply_control(control)

            # ====== 卡死/贴墙检测 → 进入恢复，但不重设终点 ======
            if (speed < RECOVER_TRIGGER_SPEED) and (abs(steer) > RECOVER_TRIGGER_STEER):
                _stuck_counter += 1
            else:
                _stuck_counter = max(0, _stuck_counter - 1)

            if _stuck_counter >= RECOVER_TRIGGER_TICKS:
                print("[Recover-soft] snap-to-lane & cooldown (no replan)")
                snap_to_lane_center(vehicle, world)  # 对正
                _recover_until = now + RECOVER_COOLDOWN_S
                _stuck_counter = 0
                # 不调用 set_agent_destination，保持当前终点

            if now - log_t > 1.0:
                log_t = now
                print(f"[State] speed={speed*3.6:5.1f} km/h throttle={control.throttle:.2f} brake={control.brake:.2f} steer={steer:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        try: world.apply_settings(old_settings)
        except: pass
        try: tm.set_synchronous_mode(False)
        except: pass
        for a in actors:
            try: a.destroy()
            except: pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
