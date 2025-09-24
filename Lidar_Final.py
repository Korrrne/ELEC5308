
# 1_merged_v2.py — Keep 1.py display; control from Lidar.py; fix nearest-point & improve marker
#
# Fixes:
# 1) Avoid choosing LiDAR concentric ground rings as "nearest obstacle":
#    - Ground/sky z-filter in LiDAR frame (below).
#    - Neighborhood density vetting (reject isolated ring pixels).
# 2) Make nearest obstacle marker obvious:
#    - Thick line from ego to point, bold circle + crosshair, text HUD (range/angle).
#
import glob, os, sys, time, math, threading, random
import numpy as np
import cv2

# ========= Modify to your CARLA PythonAPI path =========
carla_root = r"D:\CARLA9\WindowsNoEditor\PythonAPI"
sys.path.append(carla_root)
sys.path.append(os.path.join(carla_root, "carla"))
sys.path.append(os.path.join(carla_root, "agents"))
try:
    import sys as _sys
    egg = glob.glob(os.path.join(
        carla_root, "dist",
        f"carla-*{_sys.version_info.major}.{_sys.version_info.minor}-"
        f"{'win-amd64' if os.name=='nt' else 'linux-x86_64'}.egg"))
    if egg: sys.path.append(egg[0])
except Exception:
    pass

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent  # not used, kept for parity

# ========= Shared/basic parameters =========
LIDAR_RANGE_M = 25.0          # coloring normalization upper bound; keep in sync with sensor 'range'
SYNC_MODE = True
WINDOW_NAME = "BEV (Range-colored: near=red, far=blue)"

# BEV canvas
BEV_W, BEV_H = 900, 900
BEV_PX_PER_M = 18

# ========= Nearest obstacle (polar; vehicle coords: x forward, y right, z up) =========
NEAREST_R = float("inf")
NEAREST_THETA_DEG = 0.0

# Thread-safe visualization buffer
_bev_vis = None
_vis_lock = threading.Lock()

# ========= Exclude the ego rectangle (only for "nearest" search) =========
VEHICLE_EXCL_LEN_M   = 4.6   # along x (front/back)
VEHICLE_EXCL_WID_M   = 2.2   # along y (left/right)
EXCL_MARGIN_M        = 0.25  # extra margin

# LiDAR mount offset relative to vehicle geometric center (+x forward; +y right)
LIDAR_OFF_X_M        = 0.0
LIDAR_OFF_Y_M        = 0.0

# Visualization options
DRAW_EXCL_RECT_IN_BEV = True   # draw the exclusion rectangle (white box)
MASK_EXCL_RECT_IN_BEV = False  # if True, also mask that area in BEV image

# ------- Ground/Sky filtering to suppress rings -------
USE_GROUND_FILTER = True
GROUND_Z_MIN = -1.2    # meters relative to LiDAR (ground around -1.8 for 1.8m mount) -> keep points above this
GROUND_Z_MAX =  1.0    # remove high sky/trees far above sensor

# ------- Density vetting to suppress isolated ring pixels -------
USE_DENSITY_VET = True
DENSITY_RADIUS_M = 0.35      # neighborhood radius
DENSITY_THETA_DEG = 4.0      # angular neighborhood
DENSITY_MIN_NEIGHBORS = 8    # require at least N neighbors near candidate

def ensure_window():
    try:
        cv2.getWindowProperty(WINDOW_NAME, 0)
    except:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 720, 720)

# =========================================
# LiDAR callback: BEV range coloring + nearest obstacle (ego-rect excluded) + 0° baseline
# =========================================
def cb_lidar(meas: carla.LidarMeasurement):
    global NEAREST_R, NEAREST_THETA_DEG, _bev_vis

    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
    if pts.size == 0:
        return

    # Vehicle coords: x forward, y right, z up (LiDAR local frame)
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    intensity = pts[:, 3]

    # —— BEV projection (vehicle at canvas center) ——
    cx, cy = BEV_W // 2, BEV_H // 2
    px = (cx + y * BEV_PX_PER_M).astype(np.int32)
    py = (cy - x * BEV_PX_PER_M).astype(np.int32)
    mask_in = (px >= 0) & (px < BEV_W) & (py >= 0) & (py < BEV_H)

    # —— Range/angle (planar); angle 0° = forward, right positive ——
    r_all = np.sqrt(x * x + y * y)
    theta_all = np.degrees(np.arctan2(y, x))

    # —— Base candidate mask for "nearest": inside BEV & outside ego-rect ——
    base = mask_in.copy()

    # Translate points to vehicle geometric center if LiDAR is offset
    hx = VEHICLE_EXCL_LEN_M * 0.5 + EXCL_MARGIN_M
    hy = VEHICLE_EXCL_WID_M * 0.5 + EXCL_MARGIN_M
    xr = x - LIDAR_OFF_X_M
    yr = y - LIDAR_OFF_Y_M
    rect_mask = (np.abs(xr) <= hx) & (np.abs(yr) <= hy)  # True means inside ego rectangle
    base &= ~rect_mask

    # —— Ground/Sky filter ——
    if USE_GROUND_FILTER:
        gz = (z >= GROUND_Z_MIN) & (z <= GROUND_Z_MAX)
        base &= gz

    # —— Density vetting helper ——
    nearest_idx = None
    if np.any(base):
        # sort by range
        cand_idx = np.flatnonzero(base)
        order = np.argsort(r_all[cand_idx])
        cand_idx = cand_idx[order]

        if USE_DENSITY_VET:
            # vectorized precomputation for speed
            # We'll try the first K candidates and pick the first that has enough neighbors
            K = min(200, cand_idx.size)
            rr = r_all[cand_idx[:K]]
            tt = theta_all[cand_idx[:K]]
            for i in range(K):
                ri = rr[i]; ti = tt[i]
                # neighbor set within ring +/-dr and +/-dtheta among all base points
                # use all base points but short-circuit if enough found
                rdiff = np.abs(r_all[base] - ri)
                tdiff = np.abs(theta_all[base] - ti)
                # wrap-around for angles near -180/180
                tdiff = np.minimum(tdiff, 360.0 - tdiff)
                neigh = (rdiff <= DENSITY_RADIUS_M) & (tdiff <= DENSITY_THETA_DEG)
                if np.count_nonzero(neigh) >= DENSITY_MIN_NEIGHBORS:
                    nearest_idx = int(cand_idx[i])
                    break
        # fallback: if density vetting failed, still take the geometric nearest
        if nearest_idx is None and cand_idx.size > 0:
            nearest_idx = int(cand_idx[0])
            # If this is likely a ring artifact (very low density), blank it
            if USE_DENSITY_VET:
                ri = r_all[nearest_idx]; ti = theta_all[nearest_idx]
                rdiff = np.abs(r_all[base] - ri)
                tdiff = np.abs(theta_all[base] - ti)
                tdiff = np.minimum(tdiff, 360.0 - tdiff)
                if np.count_nonzero((rdiff <= DENSITY_RADIUS_M) & (tdiff <= DENSITY_THETA_DEG)) < DENSITY_MIN_NEIGHBORS:
                    nearest_idx = None

    if nearest_idx is not None:
        NEAREST_R = float(r_all[nearest_idx])
        NEAREST_THETA_DEG = float(theta_all[nearest_idx])
    else:
        NEAREST_R = float("inf")
        NEAREST_THETA_DEG = 0.0

    # —— BEV range coloring: per-pixel min range → normalize (near=1, far=0) → TURBO ——
    dist_map = np.full((BEV_H, BEV_W), np.inf, dtype=np.float32)
    np.minimum.at(dist_map, (py[mask_in], px[mask_in]), r_all[mask_in])

    norm = np.zeros_like(dist_map, dtype=np.float32)
    valid = np.isfinite(dist_map)
    if np.any(valid):
        norm[valid] = 1.0 - np.clip(dist_map[valid] / float(LIDAR_RANGE_M), 0.0, 1.0)

    bev_u8 = (norm * 255.0).astype(np.uint8)
    bev_color = cv2.applyColorMap(bev_u8, cv2.COLORMAP_TURBO)

    # —— 0° baseline (forward): from center to top edge ——
    cv2.line(bev_color, (cx, cy), (cx, 0), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(bev_color, (cx, 0), 2, (255, 255, 255), -1)

    # —— Ego exclusion rectangle visualization ——
    if DRAW_EXCL_RECT_IN_BEV or MASK_EXCL_RECT_IN_BEV:
        hx_px = int(hx * BEV_PX_PER_M)
        hy_px = int(hy * BEV_PX_PER_M)
        dx_px = int(LIDAR_OFF_Y_M * BEV_PX_PER_M)   # image x corresponds to vehicle y (right +)
        dy_px = int(LIDAR_OFF_X_M * BEV_PX_PER_M)   # image y corresponds to vehicle x (forward +; upward -)

        x1, x2 = cx - hy_px + dx_px, cx + hy_px + dx_px
        y1, y2 = cy + dy_px - hx_px, cy + dy_px + hx_px  # forward is upward, thus -hx_px

        x1 = max(0, min(BEV_W - 1, x1)); x2 = max(0, min(BEV_W - 1, x2))
        y1 = max(0, min(BEV_H - 1, y1)); y2 = max(0, min(BEV_H - 1, y2))

        if MASK_EXCL_RECT_IN_BEV:
            bev_color[y1:y2+1, x1:x2+1] = 0
        if DRAW_EXCL_RECT_IN_BEV:
            cv2.rectangle(bev_color, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    # —— Mark nearest point (bold & obvious) ——
    if np.isfinite(NEAREST_R) and NEAREST_R < 1e6:
        # compute its image coords
        nx = int(np.clip(cx + (NEAREST_R * math.sin(math.radians(NEAREST_THETA_DEG))) * BEV_PX_PER_M, 0, BEV_W - 1))
        ny = int(np.clip(cy - (NEAREST_R * math.cos(math.radians(NEAREST_THETA_DEG))) * BEV_PX_PER_M, 0, BEV_H - 1))

        # line from ego to point
        cv2.line(bev_color, (cx, cy), (nx, ny), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(bev_color, (cx, cy), (nx, ny), (255, 255, 255), 2, cv2.LINE_AA)

        # thick circle + crosshair
        cv2.circle(bev_color, (nx, ny), 8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(bev_color, (nx, ny), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(bev_color, (nx, ny), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=3)
        cv2.drawMarker(bev_color, (nx, ny), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=1)

        # text HUD near top-left
        txt = f"Nearest: {NEAREST_R:4.1f} m @ {NEAREST_THETA_DEG:+5.1f} deg"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x0, y0 = 12, 22
        cv2.rectangle(bev_color, (x0-6, y0-18), (x0+tw+6, y0+6), (0,0,0), -1)
        cv2.putText(bev_color, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    # —— Write visualization buffer ——
    with _vis_lock:
        _bev_vis = bev_color

# ======================================================
# Driving control module (from Lidar.py): BehaviorAgent
# - Speed cap ~ 8.3 km/h
# - Steering smoothing & clamping
# - Soft "snap-to-lane" recovery without changing destination
# ======================================================
DESIRED_SPEED_MPS = 2.3   # ≈ 8.3 km/h

RECOVER_COOLDOWN_S = 1.2  # hold period: brake + steer cap
RECOVER_BRAKE = 0.45
RECOVER_STEER_CAP = 0.25
RECOVER_TRIGGER_STEER = 0.75
RECOVER_TRIGGER_SPEED = 0.6  # m/s
RECOVER_TRIGGER_TICKS = 10   # with tick=0.1s, ~1s

_last_plan_time = 0.0
REPLAN_COOLDOWN_S = 2.0
_recover_until = 0.0
_stuck_counter = 0

def snap_to_lane_center(vehicle, world):
    wp = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return
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
    agent.set_destination(dest)   # BehaviorAgent: pass only the destination
    _last_plan_time = time.time()
    print("[Agent] New destination set. Dist = %.1f m" % start_loc.distance(dest))

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

# =========================================
# Main
# =========================================
def main():
    global _recover_until, _stuck_counter, _last_plan_time

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    tm = client.get_trafficmanager()

    old_settings = world.get_settings()
    if SYNC_MODE:
        new = world.get_settings()
        new.synchronous_mode = True
        new.fixed_delta_seconds = 0.1   # 10 Hz
        world.apply_settings(new)
        tm.set_synchronous_mode(True)

    actors = []
    lidar = None
    try:
        bp = world.get_blueprint_library()

        # 1) Vehicle
        cand = bp.filter('vehicle.*model3*') or bp.filter('vehicle.*mustang*')
        vehicle_bp = cand[0] if len(cand) else bp.find('vehicle.audi.tt')
        spawn_points = world.get_map().get_spawn_points()
        tf = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0, y=0, z=0.5))
        vehicle = world.try_spawn_actor(vehicle_bp, tf)
        if vehicle is None:
            vehicles = world.get_actors().filter('vehicle.*')
            if len(vehicles) == 0:
                raise RuntimeError("No vehicle available.")
            print("[Init] Attach to existing vehicle.")
            vehicle = vehicles[0]
        else:
            actors.append(vehicle)
        vehicle.set_autopilot(False)
        print(f"[Init] Vehicle ready: {vehicle.type_id}")

        snap_to_lane_center(vehicle, world)

        # 2) LiDAR
        lidar_bp = bp.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_RANGE_M))
        lidar_bp.set_attribute('rotation_frequency', '10.0')     # Hz
        lidar_bp.set_attribute('channels', '32')                 # moderate
        lidar_bp.set_attribute('points_per_second', '300000')    # moderate
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        lidar_bp.set_attribute('horizontal_fov', '360.0')
        lidar_bp.set_attribute('sensor_tick', '0.1')             # align with sync tick

        lidar_tf = carla.Transform(carla.Location(x=LIDAR_OFF_X_M, y=LIDAR_OFF_Y_M, z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        actors.append(lidar)
        print("[Init] LiDAR spawned and attached.")

        # 3) Lidar callback + window
        lidar.listen(cb_lidar)
        ensure_window()

        # 4) BehaviorAgent with speed limit
        agent = BehaviorAgent(vehicle, behavior='normal')
        agent._local_planner.set_speed(DESIRED_SPEED_MPS)
        set_agent_destination(agent, world, vehicle.get_location())

        # 5) Main loop: tick + display + driving + logs
        last_log = 0.0
        start_time = time.time()

        print("[Run] Main loop started. Press Ctrl+C to quit.")
        while True:
            if SYNC_MODE:
                world.tick()
            else:
                world.wait_for_tick()

            now = time.time()

            # Replan only outside cooldown
            if (now - _last_plan_time) >= REPLAN_COOLDOWN_S and agent.done():
                set_agent_destination(agent, world, vehicle.get_location())

            control = agent.run_step()

            # Steering clamping/smoothing
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

            # ===== Soft recovery window: brake, steer cap, keep destination =====
            if now < _recover_until:
                control.throttle = 0.0
                control.brake = max(control.brake, RECOVER_BRAKE)
                steer = float(np.clip(steer, -RECOVER_STEER_CAP, RECOVER_STEER_CAP))
                control.steer = steer
                control.manual_gear_shift = False
                vehicle.apply_control(control)
            else:
                # Normal speed governor
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

            # ===== Stuck/against-wall detection → enter recovery (no replan) =====
            if (speed < RECOVER_TRIGGER_SPEED) and (abs(steer) > RECOVER_TRIGGER_STEER):
                _stuck_counter += 1
            else:
                _stuck_counter = max(0, _stuck_counter - 1)

            if _stuck_counter >= RECOVER_TRIGGER_TICKS:
                print("[Recover-soft] snap-to-lane & cooldown (no replan)")
                snap_to_lane_center(vehicle, world)
                _recover_until = now + RECOVER_COOLDOWN_S
                _stuck_counter = 0

            # Display BEV
            with _vis_lock:
                vis = None if _bev_vis is None else _bev_vis.copy()
            if vis is not None:
                cv2.imshow(WINDOW_NAME, vis)
                cv2.waitKey(1)

            # HUD: nearest obstacle once per second
            if now - last_log > 1.0:
                last_log = now
                if math.isfinite(NEAREST_R):
                    print(f"[State] speed={speed*3.6:5.1f} km/h  nearest={NEAREST_R:4.1f} m  theta={NEAREST_THETA_DEG:+5.1f}°")
                else:
                    print(f"[State] speed={speed*3.6:5.1f} km/h  nearest=N/A")

    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C received.")
    finally:
        if SYNC_MODE:
            try: tm.set_synchronous_mode(False)
            except: pass
            try: world.apply_settings(old_settings)
            except: pass
        if lidar is not None:
            try: lidar.stop()
            except: pass
        for a in actors:
            try: a.destroy()
            except: pass
        cv2.destroyAllWindows()
        print("[Cleanup] All actors destroyed. Goodbye.")

if __name__ == "__main__":
    main()
