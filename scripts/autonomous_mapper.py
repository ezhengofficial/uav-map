import math, time, heapq, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import airsim

# -----------------------------
# Parameters (tweak as needed)
# -----------------------------
RES = 0.5                   # meters/cell
GRID_W, GRID_H = 200, 200   # 100m x 100m local map
FREE, OCC, UNK = 0, 1, -1

ALTITUDE = -12.0
CRUISE = 4.0                # m/s
LIDAR = "Lidar1"

LOG_PERIOD = 0.25           # seconds between LiDAR mapping updates
COVERAGE_TARGET = 0.60      # stop when %known cells >= 60%
MIN_KNOWN_CELLS = 1000      # don't consider coverage until we've mapped at least this many cells
MAX_CYCLES = 200
INFLATION = 1               # obstacle inflation radius in cells

SEED_SCAN_SECS = 2.0        # <-- NEW: seconds of initial scan to seed free space

# Run folder (each run gets a timestamped directory)
RUN_DIR = (Path(__file__).resolve().parents[1] / "data" / "logs" /
           datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Math / transforms
# -----------------------------
def quat_to_rotm(q):
    """AirSim quaternion -> 3x3 rotation matrix. AirSim order is (w,x,y,z)."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float32)

def transform_lidar_to_world(pts_sensor, vehicle_pose, sensor_pose):
    """
    pts_sensor: Nx3 in sensor (LiDAR) frame
    vehicle_pose: kinematics_estimated (position+orientation) from AirSim state
    sensor_pose:  ld.pose from getLidarData (pose of sensor in vehicle frame)
    returns Nx3 in world frame
    """
    if pts_sensor.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    R_vs = quat_to_rotm(sensor_pose.orientation)
    t_vs = np.array([sensor_pose.position.x_val,
                     sensor_pose.position.y_val,
                     sensor_pose.position.z_val], dtype=np.float32)

    R_wv = quat_to_rotm(vehicle_pose.orientation)
    t_wv = np.array([vehicle_pose.position.x_val,
                     vehicle_pose.position.y_val,
                     vehicle_pose.position.z_val], dtype=np.float32)

    p_v = (pts_sensor @ R_vs.T) + t_vs
    p_w = (p_v @ R_wv.T) + t_wv
    return p_w

# -----------------------------
# Mapping utils
# -----------------------------
def world_to_grid(x, y, origin_xy):
    ox, oy = origin_xy
    gx = int(round((x - ox)/RES + GRID_W/2))
    gy = int(round((y - oy)/RES + GRID_H/2))
    return gx, gy

def grid_to_world(gx, gy, origin_xy):
    ox, oy = origin_xy
    x = (gx - GRID_W/2)*RES + ox
    y = (gy - GRID_H/2)*RES + oy
    return x, y

def neighbors4(gx, gy):
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = gx+dx, gy+dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny

def bresenham(x0, y0, x1, y1):
    dx = abs(x1-x0); dy = -abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1: break
        e2 = 2*err
        if e2 >= dy: err += dy; x += sx
        if e2 <= dx: err += dx; y += sy

def inflate_obstacles(grid, r=1):
    if r <= 0: return grid
    occ = np.argwhere(grid == OCC)
    for gx, gy in occ:
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = gx+dx, gy+dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    grid[nx, ny] = max(grid[nx, ny], OCC)
    return grid

def a_star(grid, start, goal):
    if start == goal: return [start]
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    openq = [(0+h(start,goal),0,start,None)]
    seen = {}
    while openq:
        f,g,u,parent = heapq.heappop(openq)
        if u in seen: continue
        seen[u] = parent
        if u == goal:
            path=[u]
            while seen[path[-1]] is not None:
                path.append(seen[path[-1]])
            path.reverse()
            return path
        for v in neighbors4(*u):
            if grid[v] == OCC: continue
            if v in seen: continue
            heapq.heappush(openq, (g+1+h(v,goal), g+1, v, u))
    return None

def frontier_mask(grid):
    """frontier = unknown cell adjacent to free."""
    free = (grid==FREE)
    unk  = (grid==UNK)
    up = np.roll(free, 1, axis=1); up[:,0]=False
    dn = np.roll(free,-1, axis=1); dn[:,-1]=False
    lf = np.roll(free, 1, axis=0); lf[0,:]=False
    rt = np.roll(free,-1, axis=0); rt[-1,:]=False
    adj_free = up | dn | lf | rt
    return unk & adj_free

def pick_frontier(grid, current_g):
    fm = frontier_mask(grid)
    ys, xs = np.where(fm.T)
    if len(xs)==0: 
        return None
    d2 = (xs - current_g[0])**2 + (ys - current_g[1])**2
    k = np.argmin(d2)
    return (int(xs[k]), int(ys[k]))

def pick_nearest_unknown(grid, current_g):
    """Fallback when no true frontiers exist yet: nearest unknown cell."""
    ys, xs = np.where((grid.T == UNK))
    if len(xs)==0: 
        return None
    d2 = (xs - current_g[0])**2 + (ys - current_g[1])**2
    k = np.argmin(d2)
    return (int(xs[k]), int(ys[k]))

# -----------------------------
# Mapping from LiDAR (world → grid)
# -----------------------------
def insert_lidar_world(grid, pts_world, origin_xy, drone_xy_g,
                       z_min=-2.0, z_max=2.0, subsample=5):
    """Project valid world points to ground, carve freespace from drone cell, mark hits."""
    if pts_world.size == 0:
        return
    xs, ys, zs = pts_world[:,0], pts_world[:,1], pts_world[:,2]
    m = (zs > z_min) & (zs < z_max)
    xs, ys = xs[m], ys[m]

    for i in range(0, xs.shape[0], max(1, subsample)):
        gx1, gy1 = world_to_grid(xs[i], ys[i], origin_xy)
        if not (0 <= gx1 < GRID_W and 0 <= gy1 < GRID_H):
            continue
        for rx, ry in bresenham(drone_xy_g[0], drone_xy_g[1], gx1, gy1):
            if grid[rx, ry] == UNK:
                grid[rx, ry] = FREE
        grid[gx1, gy1] = OCC

# -----------------------------
# Save helpers
# -----------------------------
def save_scan_world(run_dir, vehicle, frame_idx, timestamp_ns, pts_world):
    rec = {"t": int(timestamp_ns), "vehicle": str(vehicle), "points": pts_world}
    np.save(run_dir / f"{vehicle}_{frame_idx:05d}.npy", rec, allow_pickle=True)

def save_grid_artifacts(run_dir, grid):
    np.save(run_dir / "grid.npy", grid)
    pgm = np.full((GRID_H, GRID_W), 127, dtype=np.uint8)  # (H,W) for image
    pgm[grid.T == FREE] = 255
    pgm[grid.T == OCC]  = 0
    pgm_path = run_dir / "grid.pgm"
    with open(pgm_path, "wb") as f:
        f.write(f"P5\n{GRID_W} {GRID_H}\n255\n".encode("ascii"))
        f.write(pgm.tobytes())
    print(f"[INFO] Saved grid to: {pgm_path}")

# -----------------------------
# Main
# -----------------------------
def main(vehicle="UAV1"):
    print(f"[INFO] Autonomy start on {vehicle}")
    print(f"[INFO] Run directory: {RUN_DIR}")
    c = airsim.MultirotorClient()
    c.confirmConnection()

    c.enableApiControl(True, vehicle_name=vehicle)
    c.armDisarm(True, vehicle_name=vehicle)
    c.takeoffAsync(vehicle_name=vehicle).join()
    c.moveToZAsync(ALTITUDE, 3, vehicle_name=vehicle).join()

    # map init
    grid = np.full((GRID_W, GRID_H), UNK, dtype=np.int8)

    # world origin for local map = takeoff x,y
    st0 = c.getMultirotorState(vehicle_name=vehicle)
    origin_xy = (st0.kinematics_estimated.position.x_val,
                 st0.kinematics_estimated.position.y_val)

    steps = 0
    next_map_t = time.perf_counter()
    frame_idx = 0

    # ---- SEED: do a short stationary scan so we have FREE cells & real frontiers
    print(f"[INFO] Seeding map for {SEED_SCAN_SECS}s…")
    t_seed0 = time.perf_counter()
    while time.perf_counter() - t_seed0 < SEED_SCAN_SECS:
        ld = c.getLidarData(lidar_name=LIDAR, vehicle_name=vehicle)
        if ld.point_cloud:
            pts_sensor = np.array(ld.point_cloud, dtype=np.float32).reshape(-1,3)
            st = c.getMultirotorState(vehicle_name=vehicle)
            pts_world = transform_lidar_to_world(
                pts_sensor,
                vehicle_pose=st.kinematics_estimated,
                sensor_pose=ld.pose
            )
            save_scan_world(RUN_DIR, vehicle, frame_idx, ld.time_stamp, pts_world)
            frame_idx += 1
            drone_g = world_to_grid(st.kinematics_estimated.position.x_val,
                                    st.kinematics_estimated.position.y_val,
                                    origin_xy)
            insert_lidar_world(grid, pts_world, origin_xy, drone_g)
            inflate_obstacles(grid, INFLATION)
        time.sleep(0.02)

    while steps < MAX_CYCLES:
        # 1) Sense → transform to world → SAVE scan (also map update on LOG_PERIOD)
        ld = c.getLidarData(lidar_name=LIDAR, vehicle_name=vehicle)
        if ld.point_cloud:
            pts_sensor = np.array(ld.point_cloud, dtype=np.float32).reshape(-1,3)
            st = c.getMultirotorState(vehicle_name=vehicle)
            pts_world = transform_lidar_to_world(
                pts_sensor,
                vehicle_pose=st.kinematics_estimated,
                sensor_pose=ld.pose
            )
            save_scan_world(RUN_DIR, vehicle, frame_idx, ld.time_stamp, pts_world)
            frame_idx += 1

            if time.perf_counter() >= next_map_t:
                drone_g = world_to_grid(st.kinematics_estimated.position.x_val,
                                        st.kinematics_estimated.position.y_val,
                                        origin_xy)
                insert_lidar_world(grid, pts_world, origin_xy, drone_g)
                inflate_obstacles(grid, INFLATION)
                next_map_t = time.perf_counter() + LOG_PERIOD

        # 2) Coverage check (only after some known cells exist)
        known = np.count_nonzero(grid != UNK)
        coverage = known / (GRID_W*GRID_H)
        print(f"[INFO] step={steps} known={known} coverage={coverage:.3f}")
        if known >= MIN_KNOWN_CELLS and coverage >= COVERAGE_TARGET:
            print("[INFO] coverage reached; landing")
            break

        # 3) Choose goal
        st = c.getMultirotorState(vehicle_name=vehicle)
        cur_g = world_to_grid(st.kinematics_estimated.position.x_val,
                              st.kinematics_estimated.position.y_val, origin_xy)

        goal_g = pick_frontier(grid, cur_g)
        if goal_g is None:
            # Fallback: nearest unknown (early moments or weird geometry)
            goal_g = pick_nearest_unknown(grid, cur_g)
            if goal_g is None:
                print("[INFO] no frontier/unknown left; landing")
                break

        path = a_star(grid, cur_g, goal_g)
        if not path or len(path) < 2:
            # Unblock & try again next cycle
            grid[goal_g] = FREE
            steps += 1
            continue

        # 4) Fly along a decimated path; keep sensing + mapping
        stride = max(1, int(2.0/RES))  # ~2m
        for k in range(0, len(path), stride):
            gx, gy = path[k]
            wx, wy = grid_to_world(gx, gy, origin_xy)
            fut = c.moveToPositionAsync(wx, wy, ALTITUDE, CRUISE, vehicle_name=vehicle)

            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.6:
                ld = c.getLidarData(lidar_name=LIDAR, vehicle_name=vehicle)
                if ld.point_cloud:
                    pts_sensor = np.array(ld.point_cloud, dtype=np.float32).reshape(-1,3)
                    st = c.getMultirotorState(vehicle_name=vehicle)
                    pts_world = transform_lidar_to_world(
                        pts_sensor,
                        vehicle_pose=st.kinematics_estimated,
                        sensor_pose=ld.pose
                    )
                    save_scan_world(RUN_DIR, vehicle, frame_idx, ld.time_stamp, pts_world)
                    frame_idx += 1

                    if time.perf_counter() >= next_map_t:
                        drone_g = world_to_grid(st.kinematics_estimated.position.x_val,
                                                st.kinematics_estimated.position.y_val,
                                                origin_xy)
                        insert_lidar_world(grid, pts_world, origin_xy, drone_g)
                        inflate_obstacles(grid, INFLATION)
                        next_map_t = time.perf_counter() + LOG_PERIOD
                time.sleep(0.02)

            fut.join()

        steps += 1

    # land and clean up
    c.landAsync(vehicle_name=vehicle).join()
    c.armDisarm(False, vehicle_name=vehicle)
    c.enableApiControl(False, vehicle_name=vehicle)

    # save mapping artifacts
    save_grid_artifacts(RUN_DIR, grid)

    print("[INFO] autonomous mapping complete.")
    print(f"[INFO] Scans saved in: {RUN_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="UAV1", help="AirSim vehicle name (UAV1, UAV2, ...)")
    args = ap.parse_args()
    main(vehicle=args.vehicle)
