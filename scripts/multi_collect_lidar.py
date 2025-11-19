import airsim
import time
import numpy as np
from pathlib import Path
import laspy
from datetime import datetime
import math   # <<< CHANGED: needed for GPS->ENU

# -------- Config --------
VEHICLES      = ["Drone1", "Drone2", "Drone3"]
LIDAR_NAME    = "Lidar1"
BASE_ALT      = -5.0        # base z (negative is up, NED)
SPEED         = 3.0         # m/s
NUM_STEPS     = 40
LOG_INTERVAL  = 1.0         # seconds between scans

# Per-drone offsets in NED so they fly in different regions / altitudes
# (flight pattern only, not used for alignment)
FLIGHT_OFFSETS = {
    "Drone1": (  0.0,  0.0,   0.0),  # base area, base altitude
    "Drone2": ( 20.0,  0.0,  -2.0),  # shifted east, slightly higher
    "Drone3": (  0.0, 20.0,  -4.0),  # shifted north, even higher
}
# ------------------------

# Base logs folder
SAVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "logs"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# Timestamped run folder: data/logs/20251118_163000
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = SAVE_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Multi-drone logging to: {RUN_DIR}")

# Per-drone subfolders: data/logs/<RUN_ID>/Drone1, Drone2, ...
DRONE_DIRS = {}
for name in VEHICLES:
    ddir = RUN_DIR / name
    ddir.mkdir(parents=True, exist_ok=True)
    DRONE_DIRS[name] = ddir
    print(f"[INFO]  - {name} logs -> {ddir}")

client = airsim.MultirotorClient()
client.confirmConnection()


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


# <<< NEW: GPS -> ENU conversion (relative to a reference GPS)
R_EARTH = 6378137.0  # WGS84 Earth radius in meters


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
# <<< END NEW


def create_las(out_dir: Path, frame_idx: int, points_enu_world: np.ndarray, colors: np.ndarray | None = None) -> None:
    """
    Write one frame of WORLD-frame ENU points to LAS in the given directory.
    """
    if points_enu_world.size == 0:
        return

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([
        float(np.min(points_enu_world[:, 0])),
        float(np.min(points_enu_world[:, 1])),
        float(np.min(points_enu_world[:, 2])),
    ])

    las = laspy.LasData(header)
    las.x = points_enu_world[:, 0]
    las.y = points_enu_world[:, 1]
    las.z = points_enu_world[:, 2]

    if colors is not None and len(colors) == len(points_enu_world):
        rgb16 = (np.clip(colors, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        las.red   = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue  = rgb16[:, 2]

    out_path = out_dir / f"lidar_{frame_idx:04d}.las"
    las.write(str(out_path))
    print(f"[INFO] Saved {out_path.relative_to(RUN_DIR)}  ({len(points_enu_world):,} pts)")


def calibrate_static_offsets_enu() -> dict:
    """
    Calibrate *static* ENU offsets using GPS instead of local NED.

    - Drone1's GPS at calibration time defines the ENU origin.
    - For each drone, we compute its ENU position relative to Drone1.
    - These ENU offsets are then SUBTRACTED from that drone's LiDAR points
      so that all clouds are expressed in Drone1's frame.
    """
    ref = VEHICLES[0]
    offsets_enu: dict[str, np.ndarray] = {}

    # Reference GPS (Drone1 at calibration time)
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

    return offsets_enu


def log_lidar_for_drone(vehicle_name: str, frame_idx: int, offsets_enu: dict) -> None:
    """
    Get LiDAR data for one drone, convert local NED -> ENU, then apply
    a static ENU translation so all drones align in the reference frame.

    NOTE: We assume LiDAR orientation matches the drone/world axes for now.
    """
    lidar_data = client.getLidarData(vehicle_name=vehicle_name, lidar_name=LIDAR_NAME)
    if not lidar_data.point_cloud:
        print(f"[WARN] {vehicle_name} frame {frame_idx:04d}: empty point cloud")
        return

    # Treat LiDAR points as local NED around that drone
    pts_local_ned = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    pts_local_enu = ned_to_enu(pts_local_ned)  # local ENU in drone's frame

    # Static global ENU offset relative to Drone1, computed from GPS
    offset = offsets_enu.get(vehicle_name, np.zeros(3, dtype=np.float32))

    # Shift this drone's cloud into reference (Drone1) ENU frame:
    # If Drone2 is +5m North in ENU, we subtract that so its cloud overlaps Drone1's.
    pts_world_enu = pts_local_enu + offset

    # Optional color by height in world ENU
    colors = None
    if pts_world_enu.shape[0] > 0:
        z = pts_world_enu[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        denom = max(zmax - zmin, 1e-6)
        norm = (z - zmin) / denom
        colors = np.column_stack((norm, 0.5 * np.ones_like(norm), 1.0 - norm))

    create_las(DRONE_DIRS[vehicle_name], frame_idx, pts_world_enu, colors)


def main():
    # Enable API control & arm all drones
    for name in VEHICLES:
        print(f"[INFO] Enabling API control and arming {name}...")
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    # Take off all drones
    print("[INFO] Taking off all drones...")
    takeoff_futures = [client.takeoffAsync(vehicle_name=name) for name in VEHICLES]
    for f in takeoff_futures:
        f.join()

    # Base square (in NED)
    base_square = [
        (10.0,  0.0, BASE_ALT),
        (10.0, 10.0, BASE_ALT),
        ( 0.0, 10.0, BASE_ALT),
        ( 0.0,  0.0, BASE_ALT),
    ]
    wp_idx = {name: 0 for name in VEHICLES}

    # Let drones stabilize a bit
    time.sleep(1.0)
    print("[INFO] Calibrating static ENU offsets based on GPS positions...")
    offsets_enu = calibrate_static_offsets_enu()

    print("[INFO] Starting multi-drone square flight with GPS-based static world-frame logging...")
    for step in range(NUM_STEPS):
        # Move each drone along its own offset square (in NED)
        if step % 10 == 0:
            for name in VEHICLES:
                bx, by, bz = base_square[wp_idx[name]]
                ox, oy, oz = FLIGHT_OFFSETS.get(name, (0.0, 0.0, 0.0))
                tx = bx + ox
                ty = by + oy
                tz = bz + oz
                print(f"[MOVE] {name} -> ({tx:.1f}, {ty:.1f}, {tz:.1f})")
                client.moveToPositionAsync(tx, ty, tz, SPEED, vehicle_name=name)
                wp_idx[name] = (wp_idx[name] + 1) % len(base_square)

        # Log LiDAR for each drone
        for name in VEHICLES:
            log_lidar_for_drone(name, step, offsets_enu)

        time.sleep(LOG_INTERVAL)

    # Land all drones
    print("[INFO] Landing all drones...")
    land_futures = [client.landAsync(vehicle_name=name) for name in VEHICLES]
    for f in land_futures:
        f.join()

    for name in VEHICLES:
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
        print(f"[INFO] {name} disarmed and API control released.")

    print(f"[DONE] Multi-drone LiDAR run saved under {RUN_DIR}")


if __name__ == "__main__":
    main()
