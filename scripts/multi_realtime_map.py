import airsim
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime
import laspy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop  # <<< NEW: for A* priority queue

# ========================= CONFIG =========================

VEHICLES   = ["Drone1", "Drone2", "Drone3"]
LIDAR_NAME = "Lidar1"

BASE_ALT   = -10.0    # NED (negative = up)
SPEED      = 4.0     # m/s

# Scan / update timing
SCAN_INTERVAL_SEC   = 0.5     # how often we grab LiDAR from each drone
LAS_SAVE_EVERY_N_FRAMES = 1  # per-drone LAS for debugging/logging

# Periodically save full world LiDAR snapshot (all drones merged)
WORLD_SAVE_EVERY_N_SCANS = 100  # every N main-loop iterations

# Global ENU grid (meters) around reference (Drone1 GPS)
X_MIN_ENU, X_MAX_ENU = -100.0, 100.0
Y_MIN_ENU, Y_MAX_ENU = -100.0, 100.0
CELL_SIZE            = 1.0   # m per cell

# Visualization
VIS_UPDATE_PERIOD_SEC = 3.0  # how often to refresh the 2D grid image

# Height-aware obstacle mapping
MIN_HIT_COUNT_FOR_OBSTACLE = 10      # min number of points in a cell
MIN_Z_EXTENT_FOR_OBSTACLE  = 0.3     # min vertical extent (m) to consider real structure
CLEARANCE_BELOW_DRONE_M    = 1.0     # if top is within this of drone altitude -> obstacle

# Obstacle inflation / safety bubble (in grid cells)
WALL_DILATE_ITERS     = 0   # grow "hard" obstacles by N cells
OBSTACLE_DILATE_ITERS = 0   # further inflate for safety (no-fly)

# Approximate drone flight height in ENU (Up)
DRONE_ALT_ENU_APPROX = abs(BASE_ALT)   # floor ~z=0, flight ~|BASE_ALT|

# Filter out LiDAR returns that hit drone bodies
DRONE_BODY_RADIUS_M   = 0.5    # horizontal radius around drone center to remove
DRONE_BODY_Z_MARGIN_M = 0.6    # +/- vertical band around drone altitude to remove

# Frontier planning (for a single active drones)
ACTIVE_PLANNERS          = ["Drone1", "Drone2", "Drone3"]  # which drones are allowed to move
PLAN_PERIOD_SEC     = 1.0       # how often to attempt a new plan
MAX_GRID_STEPS_PER_MOVE = 15    # how many grid cells along path per command
MIN_FRONTIER_DIST_M = 10.0       # don't pick frontiers too close to the drone
REPLAN_DIST_EPS_M   = 5.0       # only replan if we're close to previous target

# Filter out tiny frontiers that are just pinholes in known regions
# Require at least this many unknown cells in a 3x3 neighborhood
MIN_FRONTIER_UNKNOWN_NEIGHBORS = 5

# Keep different drones from "fighting" over the same frontier cluster
MIN_INTERDRONE_FRONTIER_SEP_M = 10.0

# Per-drone goal history constraint:
# A newly chosen frontier goal must not be within this radius (m)
# of any of the last N goals for that drone.
GOAL_HISTORY_RADIUS_M = 10.0   # tune: 5–20 m
GOAL_HISTORY_LEN      = 5      # how many past goals we remember

# Cap how many times we let a drone target the same frontier cell
# before declaring it "blocked" (likely impossible)
MAX_GOAL_REPEATS = 2

# WGS84 Earth radius for GPS->ENU
R_EARTH = 6378137.0

# Logs
SAVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "logs"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = SAVE_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Realtime multi-drone map run dir: {RUN_DIR}")

# Per-drone subfolders (for periodic LAS logs)
DRONE_DIRS: dict[str, Path] = {}
for name in VEHICLES:
    ddir = RUN_DIR / name
    ddir.mkdir(parents=True, exist_ok=True)
    DRONE_DIRS[name] = ddir
    print(f"[INFO]  - {name} LAS logs -> {ddir}")

# World-snapshot folder (merged clouds)
WORLD_SNAP_DIR = RUN_DIR / "world_snapshots"
WORLD_SNAP_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] World snapshots -> {WORLD_SNAP_DIR}")

# ====================== COORD HELPERS ======================

def ned_to_enu(points_ned: np.ndarray) -> np.ndarray:
    """
    Convert NED points to ENU.
    NED: [X_ned, Y_ned, Z_ned] = [N, E, Down]
    ENU: [X_enu, Y_enu, Z_enu] = [E, N, Up]
    """
    return np.column_stack((
        points_ned[:, 1],        # X_enu (East)  = Y_ned
        points_ned[:, 0],        # Y_enu (North) = X_ned
        -points_ned[:, 2],       # Z_enu (Up)    = -Z_ned
    ))


def gps_to_enu(gps: airsim.GeoPoint,
               ref: airsim.GeoPoint) -> np.ndarray:
    """
    Convert GPS (lat, lon, alt) to ENU coordinates (meters)
    relative to a reference GeoPoint (ref).
    """
    d_lat = math.radians(gps.latitude  - ref.latitude)
    d_lon = math.radians(gps.longitude - ref.longitude)

    x_east  = d_lon * math.cos(math.radians(ref.latitude)) * R_EARTH
    y_north = d_lat * R_EARTH
    z_up    = gps.altitude - ref.altitude

    return np.array([x_east, y_north, z_up], dtype=np.float32)


def calibrate_static_offsets_enu(client: airsim.MultirotorClient) -> tuple[dict, airsim.GeoPoint]:
    """
    Calibrate *static* ENU offsets using GPS.

    Returns:
      (offsets_enu, gps_ref)
    where gps_ref is the reference GeoPoint of Drone1.
    """
    ref = VEHICLES[0]
    offsets_enu: dict[str, np.ndarray] = {}

    state_ref = client.getMultirotorState(vehicle_name=ref)
    gps_ref = state_ref.gps_location
    offsets_enu[ref] = np.zeros(3, dtype=np.float32)

    print("[CALIB] Using GPS-based ENU offsets")
    print(f"[CALIB] Reference drone: {ref} GPS "
          f"lat={gps_ref.latitude:.8f}, lon={gps_ref.longitude:.8f}, alt={gps_ref.altitude:.3f}")

    for name in VEHICLES[1:]:
        st = client.getMultirotorState(vehicle_name=name)
        gps = st.gps_location
        pos_enu = gps_to_enu(gps, gps_ref)   # ENU position of this drone relative to ref
        offsets_enu[name] = pos_enu

        print(f"[CALIB] {name}: GPS lat={gps.latitude:.8f}, lon={gps.longitude:.8f}, alt={gps.altitude:.3f}, "
              f"offset_enu={pos_enu}")

    return offsets_enu, gps_ref


def get_drone_enu_from_gps(client: airsim.MultirotorClient,
                           vehicle_name: str,
                           gps_ref: airsim.GeoPoint) -> np.ndarray:
    """
    Get current drone position in ENU coordinates, using GPS + same reference as mapping.
    """
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    gps = state.gps_location
    return gps_to_enu(gps, gps_ref)  # [x_east, y_north, z_up]

# ================== GRID / MAPPING STATE ===================

# Fixed grid bounds in ENU
GRID_WIDTH  = int(np.ceil((X_MAX_ENU - X_MIN_ENU) / CELL_SIZE))
GRID_HEIGHT = int(np.ceil((Y_MAX_ENU - Y_MIN_ENU) / CELL_SIZE))
print(f"[MAP] Grid size: {GRID_WIDTH} x {GRID_HEIGHT} cells (@ {CELL_SIZE}m)")

# Coverage / height maps (persist across loop)
grid_seen = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
z_min_map = np.full((GRID_HEIGHT, GRID_WIDTH), np.inf,  dtype=np.float32)
z_max_map = np.full((GRID_HEIGHT, GRID_WIDTH), -np.inf, dtype=np.float32)
hit_count = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)

# Derived masks (updated periodically)
obstacle_mask = np.zeros_like(grid_seen, dtype=bool)
frontier_mask = np.zeros_like(grid_seen, dtype=bool)

# Cells we've decided are "bad" / unreachable frontier goals
blocked_goals_mask = np.zeros_like(grid_seen, dtype=bool)


def integrate_points_into_grid(points_enu: np.ndarray) -> None:
    """
    Incrementally update coverage + height maps from new ENU points.
    Points outside the global ENU box are ignored.
    """
    if points_enu.size == 0:
        return

    x = points_enu[:, 0]
    y = points_enu[:, 1]
    z = points_enu[:, 2]

    # Filter to within ENU bounds
    mask = (
        (x >= X_MIN_ENU) & (x <= X_MAX_ENU) &
        (y >= Y_MIN_ENU) & (y <= Y_MAX_ENU)
    )
    if not np.any(mask):
        return

    x = x[mask]
    y = y[mask]
    z = z[mask]

    ix = ((x - X_MIN_ENU) / CELL_SIZE).astype(int)
    iy = ((y - Y_MIN_ENU) / CELL_SIZE).astype(int)

    # Clip to grid
    ix = np.clip(ix, 0, GRID_WIDTH  - 1)
    iy = np.clip(iy, 0, GRID_HEIGHT - 1)

    grid_seen[iy, ix] = True
    np.minimum.at(z_min_map, (iy, ix), z)
    np.maximum.at(z_max_map, (iy, ix), z)
    np.add.at(hit_count, (iy, ix), 1)


def create_las_frame(out_dir: Path,
                     frame_idx: int,
                     points_enu: np.ndarray,
                     colors: np.ndarray | None = None) -> None:
    """
    Write one ENU-frame LiDAR snapshot to LAS in the given directory.
    (Used periodically for logging.)
    """
    if points_enu.size == 0:
        return

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([
        float(np.min(points_enu[:, 0])),
        float(np.min(points_enu[:, 1])),
        float(np.min(points_enu[:, 2])),
    ])

    las = laspy.LasData(header)
    las.x = points_enu[:, 0]
    las.y = points_enu[:, 1]
    las.z = points_enu[:, 2]

    if colors is not None and len(colors) == len(points_enu):
        rgb16 = (np.clip(colors, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        las.red   = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue  = rgb16[:, 2]

    out_path = out_dir / f"lidar_{frame_idx:04d}.las"
    las.write(str(out_path))
    # print(f"[LAS] Saved {out_path.relative_to(RUN_DIR)}  ({len(points_enu):,} pts)")


def liDAR_to_world_enu_for_drone(client: airsim.MultirotorClient,
                                 vehicle_name: str,
                                 offsets_enu: dict[str, np.ndarray],
                                 drone_positions_enu: dict[str, np.ndarray]) -> np.ndarray:
    """
    Get LiDAR data for one drone, convert local NED -> local ENU,
    then apply static ENU translation so all drones align in a common world frame.
    Also filters out points that likely hit any drone body.
    """
    lidar_data = client.getLidarData(vehicle_name=vehicle_name, lidar_name=LIDAR_NAME)
    if not lidar_data.point_cloud:
        return np.empty((0, 3), dtype=np.float32)

    pts_local_ned = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    pts_local_enu = ned_to_enu(pts_local_ned)

    offset = offsets_enu.get(vehicle_name, np.zeros(3, dtype=np.float32))
    pts_world_enu = pts_local_enu + offset

    # Remove points near drone bodies (including this one)
    pts_world_enu = filter_points_near_drones(
        pts_world_enu,
        drone_positions_enu,
        DRONE_BODY_RADIUS_M,
        DRONE_BODY_Z_MARGIN_M,
    )

    return pts_world_enu


def filter_points_near_drones(points_enu: np.ndarray,
                              drone_positions_enu: dict[str, np.ndarray],
                              radius_m: float,
                              z_margin_m: float) -> np.ndarray:
    """
    Remove LiDAR points that are likely hitting drone bodies.

    For each drone:
      - Remove points within radius_m in XY
      - AND within +/- z_margin_m of that drone's altitude (ENU z)
    """
    if points_enu.size == 0:
        return points_enu

    keep = np.ones(points_enu.shape[0], dtype=bool)

    for pos in drone_positions_enu.values():
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

        dx = points_enu[:, 0] - cx
        dy = points_enu[:, 1] - cy
        dz = points_enu[:, 2] - cz

        in_xy = (dx*dx + dy*dy) <= radius_m**2
        in_z  = (dz >= -z_margin_m) & (dz <= z_margin_m)

        # Points hitting that drone
        bad = in_xy & in_z
        keep &= ~bad

    removed = int((~keep).sum())
    if removed > 0:
        print(f"[FILTER] Removed {removed} points likely on drones.")

    return points_enu[keep]


# ---------- height-aware obstacles + frontiers ----------

def update_obstacle_mask():
    """
    Build obstacle_mask from height/density info in grid.

    - First detect "core" wall cells (tall, dense structures).
    - Then dilate them to get a slightly bigger wall region.
    - Then dilate again to get a safety no-fly zone that the planner avoids.
    """
    global obstacle_mask

    # Assume you already maintain something like:
    #   grid_count[y,x] = hit count
    #   grid_z_min[y,x], grid_z_max[y,x]
    # If not, adapt this body to your existing structure.

    # Example pattern (reuse your real conditions here):
    has_hits = hit_count >= MIN_HIT_COUNT_FOR_OBSTACLE
    z_extent = np.where(has_hits, z_max_map - z_min_map, 0.0)

    # "Core" wall candidate: enough points & tall enough
    tall_enough = z_extent >= MIN_Z_EXTENT_FOR_OBSTACLE
    z_top = z_max_map
    too_tall = z_top >= (DRONE_ALT_ENU_APPROX - CLEARANCE_BELOW_DRONE_M)
    wall_core = has_hits & tall_enough & too_tall

    # --- NEW: dilation to get thicker walls and safety zone ---
    wall_mask = dilate_mask(wall_core, WALL_DILATE_ITERS)

    # No-fly zone = inflated obstacles
    inflated = dilate_mask(wall_mask, OBSTACLE_DILATE_ITERS)

    obstacle_mask = inflated


def find_frontiers_from_seen(seen: np.ndarray) -> np.ndarray:
    """
    Frontiers: unknown cells (seen == False) that have at least one
    4-connected known neighbor (seen == True).
    """
    known = seen
    h, w = known.shape
    pad = np.pad(known, 1, mode="constant", constant_values=False)

    up    = pad[0:h,   1:w+1]
    down  = pad[2:h+2, 1:w+1]
    left  = pad[1:h+1, 0:w]
    right = pad[1:h+1, 2:w+2]

    neighbor_known = up | down | left | right
    frontiers = (~known) & neighbor_known
    return frontiers


def update_frontier_mask():
    """Recompute frontier_mask from the current grid_seen and obstacle_mask,
       filter out tiny frontiers, and ignore blocked goals."""
    global frontier_mask, grid_seen

    raw_frontiers = find_frontiers_from_seen(grid_seen)
    raw_frontiers = raw_frontiers & (~obstacle_mask)

    # Filter micro-frontiers and mark them as known
    frontier_mask, grid_seen = filter_tiny_frontiers(
        raw_frontiers,
        grid_seen,
        MIN_FRONTIER_UNKNOWN_NEIGHBORS
    )

    # Never treat blocked cells as frontiers
    frontier_mask &= ~blocked_goals_mask



def filter_tiny_frontiers(frontier_mask: np.ndarray,
                          seen_grid: np.ndarray,
                          min_unknown_neighbors: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove tiny frontier cells that are just small holes in otherwise known regions.

    We compute, for each cell, how many unknown cells are in its 3x3 neighborhood.
    Frontier cells with fewer than min_unknown_neighbors unknown neighbors are dropped.
    Those dropped cells are treated as 'known' to prevent revisiting.

    Returns:
      kept_frontiers : updated frontier mask with tiny frontiers removed
      updated_seen   : seen grid with removed frontier cells marked as known
    """
    h, w = seen_grid.shape

    # unknown = True where not seen
    unknown = ~seen_grid
    u = unknown.astype(np.int32)

    # pad for 3x3 neighborhood sum
    p = np.pad(u, pad_width=1, mode="constant", constant_values=0)

    # 3x3 sum
    local_unknown = (
        p[0:h,   0:w]   + p[0:h,   1:w+1]   + p[0:h,   2:w+2] +
        p[1:h+1, 0:w]   + p[1:h+1, 1:w+1]   + p[1:h+1, 2:w+2] +
        p[2:h+2, 0:w]   + p[2:h+2, 1:w+1]   + p[2:h+2, 2:w+2]
    )

    # Frontier cells with enough unknown neighbors
    keep_mask    = local_unknown >= min_unknown_neighbors
    kept_front   = frontier_mask & keep_mask
    removed_front = frontier_mask & (~keep_mask)

    updated_seen = seen_grid.copy()
    # Treat removed micro-frontiers as known
    updated_seen[removed_front] = True

    removed_count = int(removed_front.sum())
    if removed_count > 0:
        print(f"[MAP] Filtered out {removed_count} tiny frontier cells (filled as known).")

    return kept_front, updated_seen


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    4-connected dilation on a boolean mask.

    Each iteration grows True cells to their up/down/left/right neighbors.
    """
    if iterations <= 0:
        return mask

    h, w = mask.shape
    out = mask.copy()

    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        up    = p[0:h,   1:w+1]
        down  = p[2:h+2, 1:w+1]
        left  = p[1:h+1, 0:w]
        right = p[1:h+1, 2:w+2]
        grown = up | down | left | right
        out = out | grown

    return out



# ---------- A* path planner on grid ----------

def astar_path(obstacles: np.ndarray,
               start: tuple[int, int],
               goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """
    A* on a 4-connected grid.
    obstacles[y,x] == True means blocked.
    start, goal are (iy, ix).
    Returns list of (iy, ix) from start to goal (inclusive), or None.
    """
    h, w = obstacles.shape
    sy, sx = start
    gy, gx = goal

    if not (0 <= sx < w and 0 <= sy < h):
        return None
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    if obstacles[gy, gx]:
        return None

    open_set = []
    heappush(open_set, (0, (sy, sx)))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sy, sx): None}
    g_score = {(sy, sx): 0}

    def heuristic(y, x):
        return abs(y - gy) + abs(x - gx)

    while open_set:
        _, (cy, cx) = heappop(open_set)
        if (cy, cx) == (gy, gx):
            path = []
            cur = (cy, cx)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if obstacles[ny, nx]:
                continue

            tentative_g = g_score[(cy, cx)] + 1
            if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                g_score[(ny, nx)] = tentative_g
                f = tentative_g + heuristic(ny, nx)
                heappush(open_set, (f, (ny, nx)))
                came_from[(ny, nx)] = (cy, cx)

    return None


def plan_path_to_frontier_for_drone(drone_pos_enu_xy: np.ndarray,
                                    fm_for_plan: np.ndarray
                                    ) -> tuple[list[tuple[int, int]] | None,
                                               tuple[int, int] | None]:
    """
    Multi-goal Dijkstra-style planner:

    - fm_for_plan[y,x] == True indicates frontier cells that are valid goals
      for THIS drone (after any inter-drone separation masking).
    - obstacle_mask[y,x] == True indicates blocked cells.

    We run a single Dijkstra search from the drone's start cell and stop
    as soon as we reach ANY frontier cell whose path length corresponds to
    at least MIN_FRONTIER_DIST_M.

    Returns:
      path      : list[(iy, ix)] from start to chosen goal (inclusive), or None
      goal_cell : (iy, ix) of that chosen frontier cell, or None
    """
    h, w = grid_seen.shape

    # Convert drone ENU position to grid cell
    sx = int((drone_pos_enu_xy[0] - X_MIN_ENU) / CELL_SIZE)
    sy = int((drone_pos_enu_xy[1] - Y_MIN_ENU) / CELL_SIZE)
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    start = (sy, sx)

    if not fm_for_plan.any():
        return None, None

    # Minimum path length in cells to satisfy MIN_FRONTIER_DIST_M
    min_cells = int(MIN_FRONTIER_DIST_M / CELL_SIZE)

    # Standard Dijkstra (A* with heuristic = 0)
    obstacles = obstacle_mask  # use global obstacle mask
    open_set: list[tuple[int, tuple[int, int]]] = []
    heappush(open_set, (0, start))  # (cost, (y, x))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_set:
        cost, (cy, cx) = heappop(open_set)

        # If this is a valid frontier cell and far enough from start, we are done
        if fm_for_plan[cy, cx] and g_score[(cy, cx)] >= min_cells:
            goal = (cy, cx)
            # Reconstruct path
            path: list[tuple[int, int]] = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            print(f"[PLAN] Multi-goal path length {len(path)} to frontier cell {goal}.")
            return path, goal

        # Expand neighbors
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if obstacles[ny, nx]:
                continue

            new_cost = g_score[(cy, cx)] + 1
            if (ny, nx) not in g_score or new_cost < g_score[(ny, nx)]:
                g_score[(ny, nx)] = new_cost
                heappush(open_set, (new_cost, (ny, nx)))
                came_from[(ny, nx)] = (cy, cx)

    print("[PLAN] No reachable frontier found by multi-goal Dijkstra.")
    return None, None




# ---------- SAVE FINAL MAP (arrays + PNG) ----------

def save_final_map_and_png():
    """
    Save the final occupancy/height map to an NPZ, plus a final PNG snapshot.
    """
    out_npz = RUN_DIR / "final_map.npz"
    np.savez(
        out_npz,
        grid_seen=grid_seen,
        z_min_map=z_min_map,
        z_max_map=z_max_map,
        hit_count=hit_count,
        obstacle_mask=obstacle_mask,
        frontier_mask=frontier_mask,
        x_min_enu=X_MIN_ENU,
        y_min_enu=Y_MIN_ENU,
        cell_size=CELL_SIZE,
    )
    print(f"[SAVE] Final map arrays saved to {out_npz}")

    out_png = RUN_DIR / "final_coverage.png"
    fig.savefig(out_png, dpi=150)
    print(f"[SAVE] Final coverage PNG saved to {out_png}")

# ====================== LIVE VISUALIZATION ======================

# Visualization legend:
#   0 = unknown
#   1 = seen
#   2 = obstacle
#   3 = frontier

cmap = ListedColormap([
    "#202020",  # 0 unknown
    "#55aa55",  # 1 seen
    "#ff4444",  # 2 obstacle
    "#ffff00",  # 3 frontier
])

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

vis_img = ax.imshow(
    np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8),
    origin="lower",
    cmap=cmap,
    interpolation="nearest",
    vmin=0,
    vmax=3,  # fixed range for 0..3
)
ax.set_title("Realtime Coverage / Obstacles / Frontiers (ENU grid)")
ax.set_xlabel("X cell index")
ax.set_ylabel("Y cell index")
plt.tight_layout()
plt.show(block=False)


def update_visualization():
    """
    Build a small integer grid for visualization and update imshow:
      0 = unknown
      1 = seen
      2 = obstacle
      3 = frontier
    """
    vis = np.zeros_like(grid_seen, dtype=np.uint8)
    vis[grid_seen]      = 1
    vis[obstacle_mask]  = 2
    vis[frontier_mask]  = 3

    vis_img.set_data(vis)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

# =============================== MAIN ===============================

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Takeoff all drones (hover at BASE_ALT)
    for name in VEHICLES:
        print(f"[INFO] Enabling API control and arming {name}...")
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    print("[INFO] Taking off all drones...")
    takeoff_futs = [client.takeoffAsync(vehicle_name=name) for name in VEHICLES]
    for f in takeoff_futs:
        f.join()

    # Move each drone to its base altitude
    move_futs = [
        client.moveToZAsync(BASE_ALT, SPEED, vehicle_name=name)
        for name in VEHICLES
    ]
    for f in move_futs:
        f.join()

    time.sleep(1.0)

    # GPS-based alignment
    print("[INFO] Calibrating static ENU offsets from GPS positions...")
    offsets_enu, gps_ref = calibrate_static_offsets_enu(client)

    print("[INFO] Entering realtime LiDAR scan + map + frontier-planning loop...")
    start_time       = time.time()
    last_vis_update  = start_time
    last_plan_time   = start_time
    per_drone_frame_idx = {name: 0 for name in VEHICLES}
    world_frame_idx      = 0
    scan_counter         = 0

    # Track last commanded target for each active planner (ENU x,y)
    planner_target_enu: dict[str, np.ndarray | None] = {
        name: None for name in ACTIVE_PLANNERS
    }
    # Track last chosen frontier cell (grid indices) and repeat count per drone
    planner_goal_cell: dict[str, tuple[int, int] | None] = {
        name: None for name in ACTIVE_PLANNERS
    }
    planner_goal_repeats: dict[str, int] = {
        name: 0 for name in ACTIVE_PLANNERS
    }

    from collections import deque
    # Track last commanded target for each active planner (ENU x,y)
    planner_target_enu: dict[str, np.ndarray | None] = {
        name: None for name in ACTIVE_PLANNERS
    }
    # Track last chosen frontier cell (grid indices) and repeat count per drone
    planner_goal_cell: dict[str, tuple[int, int] | None] = {
        name: None for name in ACTIVE_PLANNERS
    }
    planner_goal_repeats: dict[str, int] = {
        name: 0 for name in ACTIVE_PLANNERS
    }

    # NEW: per-drone history of recent goals in ENU (x,y)
    planner_goal_history_enu: dict[str, deque[np.ndarray]] = {
        name: deque(maxlen=GOAL_HISTORY_LEN) for name in ACTIVE_PLANNERS
    }


    try:
        while True:
            loop_start = time.time()
            # all_pts_world_list = []

             # --- 0) Get all drone ENU positions (for self-filtering) ---
            drone_positions_enu: dict[str, np.ndarray] = {}
            for name in VEHICLES:
                drone_enu = get_drone_enu_from_gps(client, name, gps_ref)  # [x, y, z]
                drone_positions_enu[name] = drone_enu

            # --- 1) Scan all drones, integrate into grid, optionally log LAS ---
            for name in VEHICLES:
                pts_world_enu = liDAR_to_world_enu_for_drone(
                    client, name, offsets_enu, drone_positions_enu
                )

                # DEBUG early
                # if scan_counter < 20:
                #     n_pts = pts_world_enu.shape[0]
                #     if n_pts > 0:
                #         x_min, x_max = float(pts_world_enu[:, 0].min()), float(pts_world_enu[:, 0].max())
                #         y_min, y_max = float(pts_world_enu[:, 1].min()), float(pts_world_enu[:, 1].max())
                #         z_min, z_max = float(pts_world_enu[:, 2].min()), float(pts_world_enu[:, 2].max())
                #         print(f"[DEBUG] {name}: {n_pts} pts, "
                #               f"X=[{x_min:.1f},{x_max:.1f}], "
                #               f"Y=[{y_min:.1f},{y_max:.1f}], "
                #               f"Z=[{z_min:.1f},{z_max:.1f}]")
                #     else:
                #         print(f"[DEBUG] {name}: 0 LiDAR points")

                integrate_points_into_grid(pts_world_enu)

                # if pts_world_enu.size > 0:
                #     all_pts_world_list.append(pts_world_enu)

                frame_idx = per_drone_frame_idx[name]
                if frame_idx % LAS_SAVE_EVERY_N_FRAMES == 0 and pts_world_enu.size > 0:
                    z = pts_world_enu[:, 2]
                    zmin, zmax = float(z.min()), float(z.max())
                    denom = max(zmax - zmin, 1e-6)
                    norm = (z - zmin) / denom
                    colors = np.column_stack(
                        (norm, 0.5 * np.ones_like(norm), 1.0 - norm)
                    )
                    create_las_frame(DRONE_DIRS[name], frame_idx, pts_world_enu, colors)

                per_drone_frame_idx[name] = frame_idx + 1

            # Periodically write full world snapshot
            # if scan_counter % WORLD_SAVE_EVERY_N_SCANS == 0 and all_pts_world_list:
            #     pts_world_all = np.vstack(all_pts_world_list)
            #     z = pts_world_all[:, 2]
            #     zmin, zmax = float(z.min()), float(z.max())
            #     denom = max(zmax - zmin, 1e-6)
            #     norm = (z - zmin) / denom
            #     colors = np.column_stack(
            #         (norm, 0.5 * np.ones_like(norm), 1.0 - norm)
            #     )
            #     create_las_frame(WORLD_SNAP_DIR, world_frame_idx, pts_world_all, colors)
            #     world_frame_idx += 1

            # scan_counter += 1

            now = time.time()

            # --- 2) Update obstacle + frontier masks + visualization periodically ---
            if now - last_vis_update >= VIS_UPDATE_PERIOD_SEC:
                update_obstacle_mask()
                update_frontier_mask()
                update_visualization()

                seen_cells = int(grid_seen.sum())
                total_cells = grid_seen.size
                cov = seen_cells / max(total_cells, 1)
                num_obstacles = int(obstacle_mask.sum())
                num_frontiers = int(frontier_mask.sum())
                print(f"[MAP] Coverage: {cov*100:.2f}% "
                      f"({seen_cells}/{total_cells} cells), "
                      f"obstacles={num_obstacles}, frontiers={num_frontiers}")
                last_vis_update = now

            # --- 3) Frontier-based planning & motion for all ACTIVE_PLANNERS ---
            if now - last_plan_time >= PLAN_PERIOD_SEC:
                last_plan_time = now

                # If no frontiers at all, don't bother
                if not frontier_mask.any():
                    print("[PLAN] No frontiers available – nothing to plan this cycle.")
                else:
                    for planner_name in ACTIVE_PLANNERS:
                        # Get planner drone position in ENU
                        drone_enu = get_drone_enu_from_gps(client, planner_name, gps_ref)
                        drone_xy  = drone_enu[:2]

                        # If this drone already has a target, only replan once it gets close
                        target_xy = planner_target_enu[planner_name]
                        if target_xy is not None:
                            # dist_to_target = np.linalg.norm(drone_xy - target_xy)
                            # if dist_to_target > REPLAN_DIST_EPS_M:
                            #     # Still en route; skip re-planning for this drone
                            #     continue
                            # else:
                                # Close enough -> clear and pick a new target
                                planner_target_enu[planner_name] = None

                        # Build a per-drone frontier mask by excluding regions
                        # close to other drones' targets (simple task partition).
                        fm_for_plan = frontier_mask.copy()
                        # Never consider globally blocked frontier cells
                        fm_for_plan &= ~blocked_goals_mask
                        fm_for_plan &= ~obstacle_mask

                        ys_all, xs_all = np.where(fm_for_plan)
                        if len(xs_all) == 0:
                            continue

                        # Convert frontier cells to ENU centers
                        fx_all = X_MIN_ENU + (xs_all + 0.5) * CELL_SIZE
                        fy_all = Y_MIN_ENU + (ys_all + 0.5) * CELL_SIZE
                        pts_all = np.stack([fx_all, fy_all], axis=1)

                        # --- Existing: inter-drone separation using planner_target_enu[...] ---
                        for other_name in ACTIVE_PLANNERS:
                            if other_name == planner_name:
                                continue
                            other_target = planner_target_enu.get(other_name)
                            if other_target is None:
                                continue
                            diff = pts_all - other_target.reshape(1, 2)
                            dist = np.linalg.norm(diff, axis=1)
                            keep = dist >= MIN_INTERDRONE_FRONTIER_SEP_M
                            fm_new = np.zeros_like(fm_for_plan, dtype=bool)
                            fm_new[ys_all[keep], xs_all[keep]] = True
                            fm_for_plan = fm_new

                            ys_all, xs_all = np.where(fm_for_plan)
                            if len(xs_all) == 0:
                                break
                            fx_all = X_MIN_ENU + (xs_all + 0.5) * CELL_SIZE
                            fy_all = Y_MIN_ENU + (ys_all + 0.5) * CELL_SIZE
                            pts_all = np.stack([fx_all, fy_all], axis=1)

                        if not fm_for_plan.any():
                            continue

                        # --- NEW: filter out frontiers near this drone's recent goals ---
                        history = planner_goal_history_enu[planner_name]
                        if len(history) > 0:
                            # Stack history into (H,2)
                            hist_xy = np.stack(list(history), axis=0)  # ENU x,y
                            # Distances from each frontier cell to each past goal
                            # pts_all: (F,2), hist_xy: (H,2)
                            diff = pts_all[:, None, :] - hist_xy[None, :, :]
                            dist = np.linalg.norm(diff, axis=2)  # (F,H)
                            # keep frontiers that are >= radius from ALL past goals
                            min_dist = dist.min(axis=1)
                            keep = min_dist >= GOAL_HISTORY_RADIUS_M

                            fm_new = np.zeros_like(fm_for_plan, dtype=bool)
                            fm_new[ys_all[keep], xs_all[keep]] = True
                            fm_for_plan = fm_new

                        if not fm_for_plan.any():
                            # All frontiers in this drone's history radius; let it skip this cycle
                            print(f"[PLAN] {planner_name}: all candidate frontiers too close to recent goals.")
                            continue

                        path, goal_cell = plan_path_to_frontier_for_drone(drone_xy, fm_for_plan)
                        if path is None or goal_cell is None:
                            print(f"[PLAN] {planner_name}: no valid frontier path this cycle.")
                            continue

                        # --- NEW: repeat-count logic for this drone & goal ---
                        prev_goal = planner_goal_cell[planner_name]
                        if prev_goal == goal_cell:
                            planner_goal_repeats[planner_name] += 1
                        else:
                            planner_goal_cell[planner_name] = goal_cell
                            planner_goal_repeats[planner_name] = 1

                        if planner_goal_repeats[planner_name] > MAX_GOAL_REPEATS:
                            gy, gx = goal_cell
                            blocked_goals_mask[gy, gx] = True
                            print(f"[PLAN] {planner_name}: frontier {goal_cell} exceeded repeat cap "
                                f"({MAX_GOAL_REPEATS}); marking blocked.")
                            # Reset this drone's goal; it will choose something else next cycle
                            planner_goal_cell[planner_name] = None
                            planner_target_enu[planner_name] = None
                            planner_goal_repeats[planner_name] = 0
                            # Skip issuing a movement command this cycle; we'll replan next time
                            continue
                        # --- END NEW ---

                        if len(path) <= 1:
                            print(f"[PLAN] {planner_name}: path only start cell; skipping.")
                            continue

                        step_idx = min(1 + MAX_GRID_STEPS_PER_MOVE, len(path) - 1)
                        iy, ix = path[step_idx]

                        x_enu = X_MIN_ENU + (ix + 0.5) * CELL_SIZE
                        y_enu = Y_MIN_ENU + (iy + 0.5) * CELL_SIZE
                        target_xy = np.array([x_enu, y_enu], dtype=np.float32)

                        X_ned = float(y_enu)
                        Y_ned = float(x_enu)

                        print(f"[MOVE] {planner_name}: to grid (iy={iy}, ix={ix}) "
                            f"ENU=({x_enu:.2f},{y_enu:.2f}) "
                            f"NED=({X_ned:.2f},{Y_ned:.2f},{BASE_ALT:.2f})")

                        client.moveToPositionAsync(
                            X_ned, Y_ned, BASE_ALT, SPEED,
                            vehicle_name=planner_name
                        )
                        planner_target_enu[planner_name] = target_xy

                        # Record the *goal* cell center in ENU as a "visited goal" in history
                        gy, gx = goal_cell
                        goal_x_enu = X_MIN_ENU + (gx + 0.5) * CELL_SIZE
                        goal_y_enu = Y_MIN_ENU + (gy + 0.5) * CELL_SIZE
                        goal_xy_enu = np.array([goal_x_enu, goal_y_enu], dtype=np.float32)
                        planner_goal_history_enu[planner_name].append(goal_xy_enu)



            # --- 4) Sleep to maintain SCAN_INTERVAL_SEC ---
            elapsed = time.time() - loop_start
            sleep_dt = SCAN_INTERVAL_SEC - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt – exiting realtime loop.")

    # Save final map before landing/shutdown
    save_final_map_and_png()

    print("[INFO] Landing all drones...")
    land_futs = [client.landAsync(vehicle_name=name) for name in VEHICLES]
    for f in land_futs:
        f.join()

    for name in VEHICLES:
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
        print(f"[INFO] {name} disarmed and API control released.")

    print(f"[DONE] Realtime exploration run complete. Logs under {RUN_DIR}")


if __name__ == "__main__":
    main()
