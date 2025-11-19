import airsim
import time
import numpy as np
from pathlib import Path
import laspy
from datetime import datetime

# -------- Config --------
VEHICLE_NAME = "Drone1"
LIDAR_NAME   = "Lidar1"
HOVER_ALT    = -5.0        # z in AirSim (negative is up)
SPEED        = 3.0         # m/s
NUM_STEPS    = 40
LOG_INTERVAL = 1.0         # seconds between scans
# ------------------------

# Base logs folder
SAVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "logs"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# Timestamped run folder, e.g. data/logs/20251118_161530
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR  = SAVE_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Logging LAS frames to: {RUN_DIR}")

# AirSim client
c = airsim.MultirotorClient()
c.confirmConnection()
c.enableApiControl(True, VEHICLE_NAME)
c.armDisarm(True, VEHICLE_NAME)


def create_las(frame_idx: int, points_enu: np.ndarray, colors: np.ndarray | None = None) -> None:
    """
    Create a LAS file from point cloud data for this frame.

    points_enu: (N, 3) in ENU coordinates (X=E, Y=N, Z=Up)
    colors:     optional (N, 3) float in [0,1] RGB
    """
    if points_enu.size == 0:
        return

    header = laspy.LasHeader(point_format=3, version="1.2")
    # 1 cm precision
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

    out_path = RUN_DIR / f"lidar_{frame_idx:04d}.las"
    las.write(str(out_path))
    print(f"[INFO] Saved {out_path.name}  ({len(points_enu):,} pts)")


def log_lidar(frame_idx: int) -> None:
    """
    Collect and save LiDAR data as LAS for this frame.
    Converts AirSim NED -> ENU before writing.
    """
    lidar_data = c.getLidarData(vehicle_name=VEHICLE_NAME, lidar_name=LIDAR_NAME)
    if not lidar_data.point_cloud:
        print(f"[WARN] Frame {frame_idx:04d}: empty point cloud")
        return

    # AirSim gives points in NED: X=N, Y=E, Z=Down
    pts_ned = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    # Convert NED -> ENU (X_enu=E, Y_enu=N, Z_enu=Up)
    # AirSim: [X_ned, Y_ned, Z_ned] = [N, E, Down]
    # ENU:    [X_enu, Y_enu, Z_enu] = [E, N, Up]
    points_enu = np.column_stack((
        pts_ned[:, 1],        # X_enu (East)  = Y_ned
        pts_ned[:, 0],        # Y_enu (North) = X_ned
        -pts_ned[:, 2],       # Z_enu (Up)    = -Z_ned
    ))

    # Optional: colorize by height (using ENU Z for intuitiveness)
    colors = None
    if points_enu.shape[0] > 0:
        z = points_enu[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        denom = max(zmax - zmin, 1e-6)
        norm = (z - zmin) / denom
        colors = np.column_stack((norm, 0.5 * np.ones_like(norm), 1.0 - norm))

    create_las(frame_idx, points_enu, colors)


print("[INFO] Taking off...")
c.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
c.moveToZAsync(HOVER_ALT, SPEED, vehicle_name=VEHICLE_NAME).join()

# Simple square in world frame (AirSim NED frame)
wps = [
    (10,  0, HOVER_ALT),
    (10, 10, HOVER_ALT),
    ( 0, 10, HOVER_ALT),
    ( 0,  0, HOVER_ALT),
]
idx = 0

for step in range(NUM_STEPS):
    if step % 10 == 0:
        x, y, z = wps[idx]
        idx = (idx + 1) % len(wps)
        c.moveToPositionAsync(x, y, z, SPEED, vehicle_name=VEHICLE_NAME)
    log_lidar(step)
    time.sleep(LOG_INTERVAL)

c.landAsync(vehicle_name=VEHICLE_NAME).join()
c.armDisarm(False, VEHICLE_NAME)
c.enableApiControl(False, VEHICLE_NAME)
print(f"[DONE] LiDAR frames saved under {RUN_DIR}")
