import airsim
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop

# ================== CONFIG ==================

VEHICLE_NAME   = "Drone1"
LIDAR_NAME     = "Lidar1"

ALT_NED        = -5.0     # higher altitude to clear most walls/blocks
SPEED          = 5.0       # m/s
LOG_INTERVAL   = 0.05       # seconds between lidar frames
FRAMES_PER_SCAN = 15       # how many frames to capture per iteration

CELL_SIZE      = 0.5       # meters per coverage grid cell (coarser, more stable)
COVERAGE_TARGET = 0.85     # stop when ≥ 85% of bbox is covered
MAX_ITERS      = 100        # max scan→plan→move cycles

Z_MIN_FILTER   = None      # optional ENU Z filtering (e.g. 0.0)
Z_MAX_FILTER   = None      # optional ENU Z filtering (e.g. 10.0)

FLOOR_Z_ENU: float | None = None   # will be detected at runtime
FLOOR_Z_TOL      = 0.1    # meters: |z - FLOOR_Z_ENU| <= tol -> floor
FLOOR_COLOR_RGB  = (0.0, 1.0, 0.0)  # green in [0,1] RGB

# How far we move toward a frontier in one step (meters)
MAX_STEP_DIST  = 20.0

# Safety bounding box in NED (meters)
X_MIN_NED      = -150.0
X_MAX_NED      =  150.0
Y_MIN_NED      = -150.0
Y_MAX_NED      =  150.0

# Wall detection parameters (from 3D points)
MIN_WALL_POINTS  = 20      # min points in a cell to consider wall candidate
MIN_WALL_EXTENT  = 0.5     # min vertical extent (m) to be considered a wall
MIN_PASSABLE_HEIGHT = 4.0  # min height to be considered a wall (that can not be flown over)

# How far we treat around the current drone as definitely free (to escape newly grown walls)
ESCAPE_RADIUS_METERS = 1.5   # you can tune this (1–3m is reasonable)

# Frontier dilation / wall dilation
FRONTIER_DILATE_ITERS = 0
WALL_DILATE_ITERS     = 2   # expand raw wall cells a bit
OBSTACLE_DILATE_ITERS = 1   # expand walls further for safety bubble

# LiDAR dead-zone around drone (meters)
DEADZONE_RADIUS   = 3.0 * -ALT_NED     # ignore & optionally mark as seen
MIN_FRONTIER_DIST = 4.0 * -ALT_NED   # don't target frontier closer than this

# Filter out tiny frontiers that are just pinholes in known regions
# Require at least this many unknown cells in a 3x3 neighborhood
MIN_FRONTIER_UNKNOWN_NEIGHBORS = 5

# How far along the grid path we move per iteration (cells)
MAX_GRID_STEPS_PER_MOVE = 50

# Cap on how many times the same frontier cell can be selected in a row
MAX_FRONTIER_REPEATS = 10

# --- Collision / motion configs ---
COLLISION_STEP_DIST_M   = 1.0     # sub-step distance (meters)
MIN_MOVE_DIST_M         = 0.3     # don't bother moving if closer than this
COLLISION_EXTRA_SCANS   = 3       # extra LiDAR frames after a "collision"
NO_PROGRESS_THRESH_M    = 0.15    # if actual move < this -> treat as stuck


# ============================================

SAVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "logs"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = SAVE_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = RUN_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Single-drone exploration run dir: {RUN_DIR}")


def ned_to_enu(points_ned: np.ndarray) -> np.ndarray:
    """Convert NED points to ENU."""
    return np.column_stack((
        points_ned[:, 1],        # X_enu (East)  = Y_ned
        points_ned[:, 0],        # Y_enu (North) = X_ned
        -points_ned[:, 2],       # Z_enu (Up)    = -Z_ned
    ))


def create_las(out_dir: Path, frame_idx: int, points_enu: np.ndarray,
               colors: np.ndarray | None = None) -> None:
    """Write one frame of ENU points to LAS."""
    import laspy

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
    print(f"[SCAN] Saved {out_path.relative_to(RUN_DIR)} ({len(points_enu):,} pts)")


def log_lidar_frame(client: airsim.MultirotorClient, frame_idx: int) -> None:
    """Capture one LiDAR frame from the drone and save as LAS."""
    lidar_data = client.getLidarData(vehicle_name=VEHICLE_NAME,
                                     lidar_name=LIDAR_NAME)
    if not lidar_data.point_cloud:
        print(f"[WARN] Frame {frame_idx:04d}: empty point cloud")
        return

    pts_ned = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    pts_enu = ned_to_enu(pts_ned)

    colors = None
    if pts_enu.shape[0] > 0:
        z = pts_enu[:, 2]

        # Base height colormap (R-G-B style)
        zmin, zmax = float(z.min()), float(z.max())
        denom = max(zmax - zmin, 1e-6)
        norm = (z - zmin) / denom
        colors = np.column_stack((norm, 0.5 * np.ones_like(norm), 1.0 - norm))

        # Floor override: use dynamically detected FLOOR_Z_ENU
        global FLOOR_Z_ENU
        if FLOOR_Z_ENU is not None:
            floor_mask = np.abs(z - FLOOR_Z_ENU) <= FLOOR_Z_TOL
            if np.any(floor_mask):
                colors[floor_mask] = np.array(FLOOR_COLOR_RGB, dtype=np.float32)
        else:
            # Fallback safety: you forgot to call estimate_floor_z_enu()
            # Not fatal; just logs.
            print("[FLOOR] FLOOR_Z_ENU is None; floor coloring disabled for this frame.")

    create_las(RUN_DIR, frame_idx, pts_enu, colors)

    


def load_all_points_from_run() -> np.ndarray:
    """Load all LAS files under RUN_DIR, return stacked (N,3) ENU points."""
    import laspy

    files = sorted(RUN_DIR.rglob("*.las"))
    if not files:
        return np.empty((0, 3), dtype=np.float32)

    pts_all = []
    for f in files:
        las = laspy.read(f)
        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        pts = np.column_stack((x, y, z))
        pts_all.append(pts)

    pts_all = np.vstack(pts_all)

    if Z_MIN_FILTER is not None:
        pts_all = pts_all[pts_all[:, 2] >= Z_MIN_FILTER]
    if Z_MAX_FILTER is not None:
        pts_all = pts_all[pts_all[:, 2] <= Z_MAX_FILTER]

    return pts_all


def build_coverage_grid_with_walls(points_xyz: np.ndarray, cell_size: float):
    """
    Build coverage grid and wall mask from XYZ points in ENU.
      grid[y,x] = True if any points hit that cell
      wall_mask[y,x] = True for tall, dense vertical structures
    """
    if points_xyz.size == 0:
        raise ValueError("No points provided for coverage grid.")

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    eps = 1e-6
    width  = int(np.ceil((x_max - x_min) / cell_size + eps))
    height = int(np.ceil((y_max - y_min) / cell_size + eps))

    grid  = np.zeros((height, width), dtype=bool)
    count = np.zeros((height, width), dtype=np.int32)
    z_min = np.full((height, width), np.inf,  dtype=np.float32)
    z_max = np.full((height, width), -np.inf, dtype=np.float32)

    ix = ((x - x_min) / cell_size).astype(int)
    iy = ((y - y_min) / cell_size).astype(int)
    ix = np.clip(ix, 0, width  - 1)
    iy = np.clip(iy, 0, height - 1)

    grid[iy, ix] = True
    np.add.at(count, (iy, ix), 1)
    np.minimum.at(z_min, (iy, ix), z)
    np.maximum.at(z_max, (iy, ix), z)

    z_extent = np.where(count > 0, z_max - z_min, 0.0)

    wall_mask = (count >= MIN_WALL_POINTS) & (z_extent >= MIN_WALL_EXTENT)

    low_clutter = (z_extent > 0.0) & (z_extent < MIN_PASSABLE_HEIGHT)
    wall_mask[low_clutter] = False

    return grid, wall_mask, x_min, y_min, width, height

def carve_escape_bubble(obstacle_mask: np.ndarray,
                        x_min: float, y_min: float,
                        cell_size: float,
                        drone_pos_enu: np.ndarray,
                        radius_m: float) -> np.ndarray:
    """
    Clear obstacle cells in a small disk around the drone so it can escape
    if newly-dilated walls would otherwise trap it.

    Returns a modified obstacle_mask (copy).
    """
    h, w = obstacle_mask.shape
    xs = x_min + (np.arange(w)  + 0.5) * cell_size
    ys = y_min + (np.arange(h)  + 0.5) * cell_size
    Xc, Yc = np.meshgrid(xs, ys)

    dx = Xc - float(drone_pos_enu[0])
    dy = Yc - float(drone_pos_enu[1])
    dist2 = dx*dx + dy*dy

    mask = dist2 <= radius_m**2

    new_obs = obstacle_mask.copy()
    cleared = int((obstacle_mask & mask).sum())
    if cleared > 0:
        print(f"[MAP] Carving escape bubble around drone; cleared {cleared} obstacle cells.")
    new_obs[mask] = False
    return new_obs


def find_frontiers(grid: np.ndarray) -> np.ndarray:
    """Frontiers: unknown cells with at least one known neighbor."""
    h, w = grid.shape
    known = grid

    pad = np.pad(known, pad_width=1, mode="constant", constant_values=False)
    up    = pad[0:h,   1:w+1]
    down  = pad[2:h+2, 1:w+1]
    left  = pad[1:h+1, 0:w]
    right = pad[1:h+1, 2:w+2]

    neighbor_known = up | down | left | right
    frontiers = (~known) & neighbor_known
    return frontiers


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """4-connected dilation for a boolean mask."""
    if iterations <= 0:
        return mask
    f = mask.copy()
    h, w = f.shape
    for _ in range(iterations):
        padded = np.pad(f, 1, constant_values=False)
        up    = padded[0:h,   1:w+1]
        down  = padded[2:h+2, 1:w+1]
        left  = padded[1:h+1, 0:w]
        right = padded[1:h+1, 2:w+2]
        grown = up | down | left | right
        f = f | grown
    return f


def filter_tiny_frontiers(frontier_mask: np.ndarray,
                          grid: np.ndarray,
                          min_unknown_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove tiny frontier cells that are just small holes in otherwise known regions.

    We compute, for each cell, how many unknown cells are in its 3x3 neighborhood.
    Frontier cells with fewer than min_unknown_neighbors unknown neighbors are dropped.
    Those dropped cells are treated as 'known' to prevent revisiting.
    """
    h, w = grid.shape
    unknown = ~grid   # True where unknown

    # 3x3 sum over unknown mask (pure NumPy)
    u = unknown.astype(np.int32)
    p = np.pad(u, 1, constant_values=0)

    local_unknown = (
        p[0:h,   0:w]   + p[0:h,   1:w+1]   + p[0:h,   2:w+2] +
        p[1:h+1, 0:w]   + p[1:h+1, 1:w+1]   + p[1:h+1, 2:w+2] +
        p[2:h+2, 0:w]   + p[2:h+2, 1:w+1]   + p[2:h+2, 2:w+2]
    )

    # Frontier cells with enough unknown neighbors are kept
    keep = local_unknown >= min_unknown_neighbors
    kept_frontiers    = frontier_mask & keep
    removed_frontiers = frontier_mask & ~keep

    # Treat removed micro-frontiers as known
    grid_updated = grid.copy()
    grid_updated[removed_frontiers] = True

    removed_count = int(removed_frontiers.sum())
    if removed_count > 0:
        print(f"[MAP] Filtered out {removed_count} tiny frontier cells (filled as known).")

    return kept_frontiers, grid_updated


def compute_deadzone_mask(x_min: float, y_min: float,
                          width: int, height: int,
                          cell_size: float,
                          drone_pos_enu: np.ndarray) -> np.ndarray:
    """Return boolean mask of cells within DEADZONE_RADIUS of drone."""
    xs = x_min + (np.arange(width)  + 0.5) * cell_size
    ys = y_min + (np.arange(height) + 0.5) * cell_size
    Xc, Yc = np.meshgrid(xs, ys)

    dx = Xc - float(drone_pos_enu[0])
    dy = Yc - float(drone_pos_enu[1])
    dist2 = dx*dx + dy*dy
    return dist2 <= DEADZONE_RADIUS**2


def get_drone_position_enu(client: airsim.MultirotorClient) -> np.ndarray:
    """Get drone position from AirSim, in ENU coordinates."""
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    pos = state.kinematics_estimated.position
    x_ned = pos.x_val
    y_ned = pos.y_val
    x_enu = y_ned
    y_enu = x_ned
    return np.array([x_enu, y_enu], dtype=np.float32)

def estimate_floor_z_enu(client: airsim.MultirotorClient,
                         num_scans: int = 5,
                         sleep_sec: float = 0.1) -> float:
    """
    Estimate floor altitude in ENU using a few LiDAR frames.

    Strategy: collect all z-values from several scans and take a low percentile
    (e.g. 5th percentile) to be robust against outliers below the floor.
    """
    zs = []

    for i in range(num_scans):
        lidar_data = client.getLidarData(vehicle_name=VEHICLE_NAME,
                                         lidar_name=LIDAR_NAME)
        if not lidar_data.point_cloud:
            time.sleep(sleep_sec)
            continue

        pts_ned = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        pts_enu = ned_to_enu(pts_ned)
        if pts_enu.shape[0] == 0:
            time.sleep(sleep_sec)
            continue

        zs.append(pts_enu[:, 2])
        time.sleep(sleep_sec)

    if not zs:
        print("[FLOOR] WARNING: No points collected for floor estimation; defaulting to 0.0 ENU.")
        return 0.0

    zs = np.concatenate(zs)
    # Use a low percentile to avoid stray points below the real floor
    floor_z = float(np.percentile(zs, 5.0))
    print(f"[FLOOR] Estimated floor_z_enu ≈ {floor_z:.3f} m")
    return floor_z

def safe_move_with_collision_scan(
    client: airsim.MultirotorClient,
    target_X_ned: float,
    target_Y_ned: float,
    target_Z_ned: float,
    speed: float,
    frame_idx: int,
) -> tuple[int, bool]:
    """
    Move the drone toward (target_X_ned, target_Y_ned, target_Z_ned) in small steps.

    We DO NOT use AirSim's collision flag.
    Instead, we treat "no progress" (very small actual movement between steps)
    as a collision surrogate.

    If "collision" is detected:
      - back off a bit,
      - take a few extra LiDAR scans,
      - return (new_frame_idx, True).

    If movement completes without getting stuck:
      - return (frame_idx, False).
    """
    # Current NED position
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    pos   = state.kinematics_estimated.position
    cur   = np.array([pos.x_val, pos.y_val], dtype=np.float32)
    tgt   = np.array([target_X_ned, target_Y_ned], dtype=np.float32)

    dvec = tgt - cur
    dist = float(np.linalg.norm(dvec))
    if dist < MIN_MOVE_DIST_M:
        # We're basically already there
        return frame_idx, False

    direction = dvec / max(dist, 1e-6)

    # Step geometry + timeout per sub-step
    step_len  = min(COLLISION_STEP_DIST_M, dist)
    n_steps   = int(np.ceil(dist / step_len))
    base_time = step_len / max(speed, 0.1)
    timeout_sec = max(1.0, 2.0 * base_time)  # a bit generous but bounded

    # Previous "safe" position (start)
    prev_pos = cur.copy()

    for k in range(1, n_steps + 1):
        step_dist = min(k * step_len, dist)
        step_pos  = cur + direction * step_dist

        X_step, Y_step = clamp_ned_within_bbox(
            float(step_pos[0]),
            float(step_pos[1])
        )

        # Command short hop with timeout
        client.moveToPositionAsync(
            X_step, Y_step, target_Z_ned,
            speed,
            timeout_sec=timeout_sec,
            vehicle_name=VEHICLE_NAME
        ).join()

        # Get new position after the step
        state_after = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        pos_after   = state_after.kinematics_estimated.position
        new_pos = np.array([pos_after.x_val, pos_after.y_val], dtype=np.float32)

        # How far did we actually move since last safe position?
        actual_move = float(np.linalg.norm(new_pos - prev_pos))

        # --- NO-PROGRESS = "COLLISION" ---
        if actual_move < NO_PROGRESS_THRESH_M:
            print(f"[COLLISION] No-progress detected "
                  f"(moved {actual_move:.3f} m); backing off and rescanning.")

            # Back off slightly toward previous safe position
            back = prev_pos + 0.3 * (prev_pos - new_pos)
            back_X, back_Y = clamp_ned_within_bbox(float(back[0]), float(back[1]))

            client.moveToPositionAsync(
                back_X, back_Y, target_Z_ned,
                speed,
                timeout_sec=timeout_sec,
                vehicle_name=VEHICLE_NAME
            ).join()

            # Extra LiDAR frames right after "collision"
            for _ in range(COLLISION_EXTRA_SCANS):
                log_lidar_frame(client, frame_idx)
                frame_idx += 1
                time.sleep(LOG_INTERVAL)

            return frame_idx, True

        # No "collision": update prev_pos to the new position
        prev_pos = new_pos

    return frame_idx, False


def clamp_ned_within_bbox(X_ned: float, Y_ned: float) -> tuple[float, float]:
    """Clamp NED coordinates into safety bounding box."""
    X_clamped = float(np.clip(X_ned, X_MIN_NED, X_MAX_NED))
    Y_clamped = float(np.clip(Y_ned, Y_MIN_NED, Y_MAX_NED))
    return X_clamped, Y_clamped

def compute_flyable_mask(x_min: float, y_min: float,
                         width: int, height: int,
                         cell_size: float) -> np.ndarray:
    """
    Returns a boolean mask [height, width] where True means the cell center
    lies inside the allowed NED flight box [X_MIN_NED..X_MAX_NED] x [Y_MIN_NED..Y_MAX_NED].

    Grid is in ENU; conversion:
      ENU (x_enu, y_enu) -> NED:
        X_ned = y_enu
        Y_ned = x_enu
    """
    xs = x_min + (np.arange(width)  + 0.5) * cell_size
    ys = y_min + (np.arange(height) + 0.5) * cell_size
    Xc, Yc = np.meshgrid(xs, ys)   # ENU centers

    # ENU -> NED
    X_ned = Yc
    Y_ned = Xc

    flyable = (
        (X_ned >= X_MIN_NED) & (X_ned <= X_MAX_NED) &
        (Y_ned >= Y_MIN_NED) & (Y_ned <= Y_MAX_NED)
    )
    return flyable


def save_grid_figure(grid: np.ndarray,
                     frontier_mask: np.ndarray,
                     wall_mask: np.ndarray,
                     obstacle_mask: np.ndarray,
                     x_min: float, y_min: float,
                     cell_size: float,
                     drone_pos_enu: np.ndarray | None,
                     iteration: int) -> None:
    """
    Save a PNG showing coverage + frontiers + walls + do-not-fly.
      0 = unknown
      1 = known
      2 = frontier
      3 = wall
      4 = do-not-fly-through (inflated walls)
    """
    vis = np.zeros_like(grid, dtype=np.uint8)
    vis[grid] = 1
    vis[frontier_mask] = 2
    vis[wall_mask] = 3
    vis[obstacle_mask] = 4

    cmap = ListedColormap([
        "#202020",  # 0 unknown
        "#55aa55",  # 1 known
        "#ff4444",  # 2 frontier
        "#3366ff",  # 3 wall
        "#ffcc00",  # 4 do-not-fly
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(vis, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(f"Coverage / Frontiers / Walls (iter {iteration})")
    ax.set_xlabel("X cell index")
    ax.set_ylabel("Y cell index")

    if drone_pos_enu is not None:
        x_enu, y_enu = float(drone_pos_enu[0]), float(drone_pos_enu[1])
        ix = (x_enu - x_min) / cell_size
        iy = (y_enu - y_min) / cell_size
        ax.scatter(ix, iy, s=40, marker="x", color="cyan", label="Drone")
        ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["Unknown", "Known", "Frontier", "Wall", "No-fly"])

    fig.tight_layout()
    out_path = DEBUG_DIR / f"coverage_iter_{iteration:02d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DEBUG] Saved coverage visualization: {out_path}")


# ---------- NEW: grid A* path planning over obstacle_mask ----------

def astar_path(obstacle_mask: np.ndarray,
               start: tuple[int, int],
               goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """
    A* search on a 4-connected grid.

    obstacle_mask[y,x] == True means blocked.
    start, goal are (iy, ix).
    Returns list of (iy, ix) from start to goal (inclusive), or None.
    """
    h, w = obstacle_mask.shape
    sy, sx = start
    gy, gx = goal

    if not (0 <= sx < w and 0 <= sy < h):
        return None
    if not (0 <= gx < w and 0 <= gy < h):
        return None

    # allow starting in an obstacle cell (we're already there), but not entering others
    if obstacle_mask[gy, gx]:
        return None

    open_set = []
    heappush(open_set, (0, (sy, sx)))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sy, sx): None}
    g_score = { (sy, sx): 0 }

    def heuristic(y, x):
        return abs(y - gy) + abs(x - gx)  # Manhattan

    while open_set:
        _, (cy, cx) = heappop(open_set)
        if (cy, cx) == (gy, gx):
            # reconstruct path
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
            if obstacle_mask[ny, nx]:
                continue
            tentative_g = g_score[(cy, cx)] + 1
            if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                g_score[(ny, nx)] = tentative_g
                f = tentative_g + heuristic(ny, nx)
                heappush(open_set, (f, (ny, nx)))
                came_from[(ny, nx)] = (cy, cx)

    return None


def plan_path_to_frontier(frontier_mask: np.ndarray,
                          obstacle_mask: np.ndarray,
                          x_min: float, y_min: float,
                          cell_size: float,
                          drone_pos_enu: np.ndarray,
                          blocked_cells: set[tuple[int, int]] | None = None,
                          max_candidates: int = 100
                          ) -> tuple[list[tuple[int, int]] | None, tuple[int, int] | None]:
    """
    Pick a frontier cell that:
      - is not in obstacle_mask
      - is not too close to the drone
      - is not in blocked_cells (frontiers we've given up on)
      - is reachable via A* on the grid

    Returns (path, goal_cell) where:
        path      : list of (iy, ix) from start to goal (inclusive), or None
        goal_cell : (iy, ix) of the chosen frontier, or None if no path
    """
    h, w = obstacle_mask.shape

    # drone start cell
    sx = int((drone_pos_enu[0] - x_min) / cell_size)
    sy = int((drone_pos_enu[1] - y_min) / cell_size)
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    start = (sy, sx)

    ys, xs = np.where(frontier_mask & ~obstacle_mask)
    if len(xs) == 0:
        return None

    x_enu = x_min + (xs + 0.5) * cell_size
    y_enu = y_min + (ys + 0.5) * cell_size
    pts = np.stack([x_enu, y_enu], axis=1)

    diff = pts - drone_pos_enu.reshape(1, 2)
    dist = np.linalg.norm(diff, axis=1)

    # filter out frontiers that are too close
    valid = dist >= MIN_FRONTIER_DIST
    if not np.any(valid):
        return None, None

    idxs = np.where(valid)[0]
    # sort candidate frontiers by distance
    idxs = idxs[np.argsort(dist[valid])]

    blocked_cells = blocked_cells or set()

    # try up to max_candidates nearest frontiers
    for idx in idxs[:max_candidates]:
        gy, gx = ys[idx], xs[idx]
        goal = (gy, gx)

        if goal in blocked_cells:
            continue

        path = astar_path(obstacle_mask, start, goal)
        if path is not None:
            print(f"[PLAN] Found path of length {len(path)} to frontier cell {goal}.")
            return path, goal

    print("[PLAN] No reachable frontier cells found with A* (given current blocks).")
    return None, None


# ------------------------------------------------------------------


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    print(f"[INFO] Enabling API control and arming {VEHICLE_NAME}...")
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)

    print(f"[INFO] Taking off {VEHICLE_NAME}...")
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
    client.moveToZAsync(ALT_NED, SPEED, vehicle_name=VEHICLE_NAME).join()

    global FLOOR_Z_ENU
    FLOOR_Z_ENU = estimate_floor_z_enu(client)

    frame_idx = 0
    coverage = 0.0

    # Track repeated goals
    blocked_goals: set[tuple[int, int]] = set()
    last_goal_cell: tuple[int, int] | None = None
    goal_repeat_count: int = 0

    for it in range(MAX_ITERS):
        print(f"\n===== ITERATION {it+1}/{MAX_ITERS} =====")

        # 1) Scan phase
        for _ in range(FRAMES_PER_SCAN):
            log_lidar_frame(client, frame_idx)
            frame_idx += 1
            time.sleep(LOG_INTERVAL)

        # 2) Build coverage + walls
        pts_all = load_all_points_from_run()
        if pts_all.shape[0] == 0:
            print("[WARN] No points yet; continuing...")
            continue

        grid, wall_mask_raw, x_min, y_min, width, height = build_coverage_grid_with_walls(
            pts_all, CELL_SIZE
        )

        # Cells where the drone is actually allowed to fly
        flyable_mask = compute_flyable_mask(x_min, y_min, width, height, CELL_SIZE)

        # Only count coverage inside the flyable region
        seen_cells  = int((grid & flyable_mask).sum())
        total_cells = int(flyable_mask.sum())
        coverage = seen_cells / max(total_cells, 1)
        print(f"[MAP] Grid: {width} x {height}, coverage (flyable region) = {coverage*100:.2f}%")
        print(f"[MAP] Flyable cells: {int(flyable_mask.sum())}, total cells: {grid.size}")


        # 3) Frontier detection & dilation
        frontier_mask = find_frontiers(grid)
        print(f"[MAP] Raw frontier cells: {int(frontier_mask.sum())}")
        frontier_mask = dilate_mask(frontier_mask, FRONTIER_DILATE_ITERS)
        print(f"[MAP] Dilated frontier cells: {int(frontier_mask.sum())}")

        # 4) Walls & obstacles (do-not-fly-through)
        wall_mask = dilate_mask(wall_mask_raw, WALL_DILATE_ITERS)
        obstacle_mask = dilate_mask(wall_mask, OBSTACLE_DILATE_ITERS)
        # 4a) Treat out-of-bounds cells as hard obstacles for planning
        nav_obstacle_mask = obstacle_mask | (~flyable_mask)


        # 5) Get drone pose
        drone_pos_enu = get_drone_position_enu(client)

        # 5a) Carve a small escape bubble around the drone so newly-dilated
        #     walls don't trap it inside a "forbidden" region.
        nav_obstacle_mask = carve_escape_bubble(
            nav_obstacle_mask,
            x_min, y_min,
            CELL_SIZE,
            drone_pos_enu,
            radius_m=ESCAPE_RADIUS_METERS,
        )

        # 5b) Dead-zone around drone (mask out nearby frontiers)
        deadzone_mask = compute_deadzone_mask(x_min, y_min, width, height,
                                              CELL_SIZE, drone_pos_enu)

        # Don't consider frontiers in no-fly / wall / deadzone
        frontier_mask &= ~nav_obstacle_mask
        frontier_mask &= ~deadzone_mask

        # Optionally count deadzone as seen so we don't chase the donut
        grid[deadzone_mask] = True


        # 6) Filter out tiny frontier "pinholes" and treat them as known
        frontier_mask, grid = filter_tiny_frontiers(
            frontier_mask, grid,
            min_unknown_neighbors=MIN_FRONTIER_UNKNOWN_NEIGHBORS
        )

        num_frontiers = int(frontier_mask.sum())
        print(f"[MAP] Valid frontier cells (after filters): {num_frontiers}")

        # Save debug PNG
        save_grid_figure(grid, frontier_mask, wall_mask, nav_obstacle_mask,
                         x_min, y_min, CELL_SIZE, drone_pos_enu,
                         iteration=it+1)

        if coverage >= COVERAGE_TARGET:
            print(f"[STOP] Coverage target reached ({coverage*100:.2f}%).")
            break
        if num_frontiers == 0:
            print("[STOP] No frontiers available; map may be complete.")
            break

         # 7) Plan grid path to some reachable frontier, avoiding obstacles
        path, goal_cell = plan_path_to_frontier(
            frontier_mask, nav_obstacle_mask,
            x_min, y_min, CELL_SIZE,
            drone_pos_enu,
            blocked_cells=blocked_goals,
            max_candidates=100
        )


        if path is None or goal_cell is None:
            print("[WARN] No reachable frontier with obstacle-respecting path. Stopping.")
            break

        # --- NEW: cap how many times the same frontier can be chosen in a row ---
        if last_goal_cell == goal_cell:
            goal_repeat_count += 1
        else:
            last_goal_cell = goal_cell
            goal_repeat_count = 1

        if goal_repeat_count > MAX_FRONTIER_REPEATS:
            print(f"[PLAN] Frontier {goal_cell} exceeded repeat cap "
                  f"({MAX_FRONTIER_REPEATS}); blocking and replanning.")
            blocked_goals.add(goal_cell)

            # try to replan to a different frontier this iteration
            path, goal_cell = plan_path_to_frontier(
                frontier_mask, nav_obstacle_mask,
                x_min, y_min, CELL_SIZE,
                drone_pos_enu,
                blocked_cells=blocked_goals,
                max_candidates=100
            )

            if path is None or goal_cell is None:
                print("[WARN] No alternative reachable frontier after blocking repeated one. Stopping.")
                break

            # reset repeat counter for the new goal
            last_goal_cell = goal_cell
            goal_repeat_count = 1
        # ----------------------------------------------------------------------

        # path is list of (iy, ix) from start to goal; choose a few steps ahead
        if len(path) <= 1:
            print("[PLAN] Path has only start cell; nothing to do. Stopping.")
            break

        step_index = min(1 + MAX_GRID_STEPS_PER_MOVE, len(path) - 1)
        iy, ix = path[step_index]

        # Convert chosen grid cell to ENU center
        x_enu_step = x_min + (ix + 0.5) * CELL_SIZE
        y_enu_step = y_min + (iy + 0.5) * CELL_SIZE

        # ENU -> NED
        X_ned = float(y_enu_step)
        Y_ned = float(x_enu_step)
        X_ned, Y_ned = clamp_ned_within_bbox(X_ned, Y_ned)

        print(f"[PLAN] Moving along path to grid cell (iy={iy}, ix={ix}) "
            f"-> ENU=({x_enu_step:.2f}, {y_enu_step:.2f}) "
            f"-> NED=({X_ned:.2f}, {Y_ned:.2f}, {ALT_NED:.2f})")

        frame_idx, collided = safe_move_with_collision_scan(
            client,
            target_X_ned=X_ned,
            target_Y_ned=Y_ned,
            target_Z_ned=ALT_NED,
            speed=SPEED,
            frame_idx=frame_idx,
        )

        if collided:
            # Already backed off + scanned; let next iteration rebuild map & replan
            continue


    print(f"[INFO] Landing {VEHICLE_NAME}...")
    client.landAsync(vehicle_name=VEHICLE_NAME).join()
    client.armDisarm(False, vehicle_name=VEHICLE_NAME)
    client.enableApiControl(False, vehicle_name=VEHICLE_NAME)
    print(f"[DONE] Exploration run complete. Logs in {RUN_DIR}")
    print(f"[DEBUG] Coverage PNGs saved in {DEBUG_DIR}")


if __name__ == "__main__":
    main()