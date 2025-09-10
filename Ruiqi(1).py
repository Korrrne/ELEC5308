#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-drive with CARLA agents; mount front/side cameras + center LIDAR.
Overlay colorized LIDAR on the front RGB (near=red, far=blue),
and render a bird's-eye top-down LIDAR view of nearby obstacles.
"""

from __future__ import annotations
import os, sys, glob, time, math, argparse
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Add CARLA egg to sys.path (same pattern as CARLA examples)
# ----------------------------------------------------------------------
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except Exception:
    pass

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

# =========================
# Global shared buffers
# =========================
LIDAR_RANGE_M = 50.0
_front_intr = None         # (fx, fy, cx, cy)
_latest_lidar = None       # Nx4 (x,y,z,intensity) in LIDAR local frame
_cam_front = None
_lidar = None
_headless = False
W_FRONT, H_FRONT = 800, 600
W_SIDE, H_SIDE = 800, 600

# Bird’s-eye canvas config
TOP_W, TOP_H = 700, 700
PX_PER_M = 7.0  # pixels per meter for top view


# ------------------ Utils ------------------

def _to_bgr(image: carla.Image, w: int, h: int) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
    return arr

def _intrinsics_from_fov(width: int, height: int, fov_deg: float):
    fov = math.radians(fov_deg)
    fx = (width/2) / math.tan(fov/2)
    fy = fx
    cx, cy = width/2, height/2
    return fx, fy, cx, cy

def _color_by_distance(d: np.ndarray, dmax: float) -> np.ndarray:
    """
    Map distance to BGR color where near=red, far=blue.
    Linear blend: t=d/dmax -> B=t*255, R=(1-t)*255, G=0
    """
    t = np.clip(d / max(dmax, 1e-6), 0.0, 1.0)
    R = ((1.0 - t) * 255.0).astype(np.uint8)
    G = np.zeros_like(R, dtype=np.uint8)
    B = (t * 255.0).astype(np.uint8)
    return np.stack([B, G, R], axis=-1)  # BGR

def _project_lidar_to_front_image(points_xyz: np.ndarray,
                                  intr: tuple[float,float,float,float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Project LIDAR points (in the camera's local frame!) to image pixels.
    CARLA convention: optical axis ~ +X; image u to right (Y), v downward (-Z).
    Input points_xyz: Nx3 in CAMERA frame.
    Return: (uv integer pixels Nx2, depth X>0 Nx1)
    """
    fx, fy, cx, cy = intr
    X = points_xyz[:, 0]
    Y = points_xyz[:, 1]
    Z = points_xyz[:, 2]
    front_mask = X > 0.1
    X, Y, Z = X[front_mask], Y[front_mask], Z[front_mask]
    u = (cx + fx * (Y / X)).astype(np.int32)
    v = (cy - fy * (Z / X)).astype(np.int32)
    uv = np.stack([u, v], axis=1)
    depth = np.sqrt(X*X + Y*Y + Z*Z)
    # clip to image bounds
    in_img = (u >= 0) & (u < W_FRONT) & (v >= 0) & (v < H_FRONT)
    return uv[in_img], depth[in_img]

def _lidar_to_camera_frame(points_lidar: np.ndarray,
                           lidar_tf: carla.Transform,
                           cam_tf: carla.Transform) -> np.ndarray:
    """
    Transform LIDAR local points to CAMERA local frame.
    Build 4x4 matrices from carla.Transform (R|t), then X_cam = T_cam^-1 * T_lidar * X_lidar.
    """
    def T_from_tf(tf: carla.Transform):
        loc = tf.location
        rot = tf.rotation  # degrees
        cy, sy = math.cos(math.radians(rot.yaw)), math.sin(math.radians(rot.yaw))
        cp, sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
        cr, sr = math.cos(math.radians(rot.roll)), math.sin(math.radians(rot.roll))
        # UE order: yaw (Z), pitch (Y), roll (X) — build Rz(yaw)*Ry(pitch)*Rx(roll)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]])
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]])
        Rx = np.array([[1, 0,  0],
                       [0, cr, -sr],
                       [0, sr,  cr]])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [loc.x, loc.y, loc.z]
        return T

    Tl = T_from_tf(lidar_tf)
    Tc = T_from_tf(cam_tf)
    Tc_inv = np.linalg.inv(Tc)
    # augment lidar points
    N = points_lidar.shape[0]
    pts_h = np.ones((N, 4), dtype=np.float32)
    pts_h[:, :3] = points_lidar[:, :3]
    # world->cam: Tc_inv @ Tl @ pts
    pts_cam_h = (Tc_inv @ (Tl @ pts_h.T)).T
    return pts_cam_h[:, :3]


# ------------------ Callbacks ------------------

def cb_front(image: carla.Image, cfg):
    global _front_intr, _latest_lidar, _cam_front, _lidar
    frame = _to_bgr(image, W_FRONT, H_FRONT)

    # lazily cache intrinsics
    if _front_intr is None:
        _front_intr = _intrinsics_from_fov(W_FRONT, H_FRONT, cfg['fov_front'])

    # overlay lidar if available
    try:
        if _latest_lidar is not None and _cam_front is not None and _lidar is not None:
            # get latest transforms
            lidar_tf = _lidar.get_transform()
            cam_tf = _cam_front.get_transform()

            # transform lidar->camera frame
            pts_cam = _lidar_to_camera_frame(_latest_lidar[:, :3], lidar_tf, cam_tf)

            # project to pixels
            uv, d = _project_lidar_to_front_image(pts_cam, _front_intr)

            # color near->far = red->blue
            colors = _color_by_distance(d, LIDAR_RANGE_M)

            # draw tiny circles
            for (u, v), (b, g, r) in zip(uv, colors):
                cv2.circle(frame, (int(u), int(v)), 2, (int(b), int(g), int(r)), -1)
    except Exception as e:
        # 防御性：投影过程中偶发矩阵/空数组问题不影响主循环
        # print(f"[LIDAR overlay warning] {e}")
        pass

    if not _headless:
        cv2.imshow('Front RGB + LIDAR overlay', frame)
        cv2.waitKey(1)

def cb_left_down(image: carla.Image, cfg):
    if not _headless:
        bgr = _to_bgr(image, W_SIDE, H_SIDE)
        cv2.imshow('Left Downward', bgr); cv2.waitKey(1)

def cb_right_down(image: carla.Image, cfg):
    if not _headless:
        bgr = _to_bgr(image, W_SIDE, H_SIDE)
        cv2.imshow('Right Downward', bgr); cv2.waitKey(1)

def cb_lidar(lidar_measurement: carla.LidarMeasurement):
    """
    Cache raw lidar points in LIDAR local frame. Also draw a bird’s-eye view canvas.
    """
    global _latest_lidar
    pts = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    _latest_lidar = pts  # (x,y,z,intensity)

    # Build top-down visualization in vehicle/LIDAR frame (x forward, y right)
    if _headless:
        return

    canvas = np.zeros((TOP_H, TOP_W, 3), dtype=np.uint8)
    cx, cy = TOP_W // 2, TOP_H // 2

    xy = pts[:, :2]
    d = np.linalg.norm(pts[:, :3], axis=1)
    colors = _color_by_distance(d, LIDAR_RANGE_M)

    # keep points within a square [-R,+R]
    R = min(TOP_W, TOP_H) / (2*PX_PER_M)  # in meters
    mask = (np.abs(xy[:, 0]) <= R) & (np.abs(xy[:, 1]) <= R)
    xy = xy[mask]; cols = colors[mask]

    # map meters -> pixels; x forward up, y right to the right
    px = (cx + (xy[:, 1] * PX_PER_M)).astype(np.int32)
    py = (cy - (xy[:, 0] * PX_PER_M)).astype(np.int32)

    in_bounds = (px >= 0) & (px < TOP_W) & (py >= 0) & (py < TOP_H)
    px, py, cols = px[in_bounds], py[in_bounds], cols[in_bounds]

    canvas[py, px] = cols

    # draw vehicle rectangle (approx) at center
    veh_w, veh_l = 1.9, 4.5  # meters
    w2, l2 = int(veh_w/2 * PX_PER_M), int(veh_l/2 * PX_PER_M)
    cv2.rectangle(canvas, (cx - w2, cy - l2), (cx + w2, cy + l2), (0, 255, 255), 2)
    cv2.putText(canvas, "Top-down LIDAR (near=red, far=blue)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.line(canvas, (cx, cy), (cx, cy - int(5*PX_PER_M)), (0,255,0), 2)  # forward axis

    cv2.imshow('LIDAR Bird’s-eye', canvas)
    cv2.waitKey(1)


# ------------------ Main ------------------

def main():
    global _cam_front, _lidar, _headless, W_FRONT, H_FRONT, W_SIDE, H_SIDE

    parser = argparse.ArgumentParser(description='CARLA autopilot + cameras + LIDAR overlay')
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--agent', choices=['Behavior', 'Basic'], default='Behavior')
    parser.add_argument('--behavior', choices=['cautious', 'normal', 'aggressive'], default='normal')
    parser.add_argument('--model', default='vehicle.*')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--fov_front', type=float, default=90.0)
    parser.add_argument('--fov_side', type=float, default=70.0)
    parser.add_argument('--res_front', type=str, default='800x600')
    parser.add_argument('--res_side', type=str, default='800x600')
    parser.add_argument('--sensor_hz', type=float, default=20.0)
    args = parser.parse_args()

    _headless = args.headless
    W_FRONT, H_FRONT = [int(x) for x in args.res_front.split('x')]
    W_SIDE,  H_SIDE  = [int(x) for x in args.res_side.split('x')]

    cfg = dict(
        headless=_headless,
        fov_front=args.fov_front,
        fov_side=args.fov_side
    )

    client = carla.Client(args.host, args.port); client.set_timeout(60.0)
    world = client.get_world()
    tm = client.get_trafficmanager()
    original_settings = world.get_settings()

    actors = []
    try:
        # sync mode = recommended
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / max(args.sensor_hz, 1.0)
            world.apply_settings(settings)
            tm.set_synchronous_mode(True)

        bp_lib = world.get_blueprint_library()

        # spawn ego vehicle
        v_bps = bp_lib.filter(args.model)
        if not v_bps: raise RuntimeError('No vehicle blueprints match filter.')
        ego_bp = np.random.choice(v_bps)
        ego_bp.set_attribute('role_name', 'hero')
        spawn = np.random.choice(world.get_map().get_spawn_points())
        vehicle = world.try_spawn_actor(ego_bp, spawn)
        if vehicle is None:
            for _ in range(20):
                spawn = np.random.choice(world.get_map().get_spawn_points())
                vehicle = world.try_spawn_actor(ego_bp, spawn)
                if vehicle: break
        if vehicle is None: raise RuntimeError('Failed to spawn vehicle.')
        actors.append(vehicle)

        tick = 1.0 / max(args.sensor_hz, 1.0)

        # ---- Cameras ----
        cam_bp = bp_lib.find('sensor.camera.rgb')

        def mk_cam_bp(width, height, fov):
            bp = bp_lib.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(width))
            bp.set_attribute('image_size_y', str(height))
            bp.set_attribute('fov', str(float(fov)))
            bp.set_attribute('sensor_tick', f'{tick:.6f}')
            return bp

        # Front (slight down pitch)
        front_bp = mk_cam_bp(W_FRONT, H_FRONT, args.fov_front)
        front_tf = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.6),
                                   carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0))
        _cam_front = world.spawn_actor(front_bp, front_tf, attach_to=vehicle)
        actors.append(_cam_front)
        _cam_front.listen(lambda data: cb_front(data, cfg))

        # Left downward
        left_bp = mk_cam_bp(W_SIDE, H_SIDE, args.fov_side)
        left_tf = carla.Transform(carla.Location(x=1.2, y=-0.9, z=1.2),
                                  carla.Rotation(pitch=-55.0, yaw=-90.0, roll=0.0))
        cam_left = world.spawn_actor(left_bp, left_tf, attach_to=vehicle); actors.append(cam_left)
        cam_left.listen(lambda data: cb_left_down(data, cfg))

        # Right downward
        right_bp = mk_cam_bp(W_SIDE, H_SIDE, args.fov_side)
        right_tf = carla.Transform(carla.Location(x=1.2, y=0.9, z=1.2),
                                   carla.Rotation(pitch=-55.0, yaw=90.0, roll=0.0))
        cam_right = world.spawn_actor(right_bp, right_tf, attach_to=vehicle); actors.append(cam_right)
        cam_right.listen(lambda data: cb_right_down(data, cfg))

        # ---- LIDAR (center of vehicle roof; you can adjust) ----
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_RANGE_M))
        lidar_bp.set_attribute('rotation_frequency', str(max(args.sensor_hz, 10.0)))
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '120000')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        lidar_bp.set_attribute('sensor_tick', f'{tick:.6f}')

        # “车的中间”：放在车顶中心更不易被遮挡；也可改 z 到 1.6 与前视相机共位
        lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8),
                                   carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        _lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        actors.append(_lidar)
        _lidar.listen(cb_lidar)

        # ---- Agent (auto driving) ----
        if args.agent == 'Behavior':
            agent = BehaviorAgent(vehicle, behavior=args.behavior)
        else:
            agent = BasicAgent(vehicle, target_speed=30)
            agent.follow_speed_limits(True)

        spawns = world.get_map().get_spawn_points()
        agent.set_destination(np.random.choice(spawns).location)

        print('[Info] Running. Windows: Front+LIDAR overlay / Left-Down / Right-Down / LIDAR Bird’s-eye')
        print('[Info] Ctrl+C to stop.')

        while True:
            if args.sync: world.tick()
            else: world.wait_for_tick()

            if agent.done():
                agent.set_destination(np.random.choice(spawns).location)
                print('[Info] Destination reached. New target set.')

            control = agent.run_step()
            control.manual_gear_shift = False
            vehicle.apply_control(control)

    except KeyboardInterrupt:
        print('\n[User] Stop requested.')

    finally:
        try:
            world.apply_settings(original_settings); tm.set_synchronous_mode(False)
        except Exception:
            pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        for a in actors[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        print('[Info] Cleanup done.')


if __name__ == '__main__':
    main()
