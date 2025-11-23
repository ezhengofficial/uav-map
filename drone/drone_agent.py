"""
Enhanced DroneAgent with COMPLETELY FIXED GPS-based world alignment.
All drones save points in a shared world frame using GPS calibration.

COMPLETE FIX: Proper GPS-to-world coordinate transformation
"""
import os
import time
import math
import numpy as np
import laspy
import airsim
from pathlib import Path
from typing import Optional, Tuple
from files_path import DATA_DIR
from scipy.spatial.transform import Rotation as R

# WGS84 Earth radius
R_EARTH = 6378137.0


class DroneAgent:
    """
    Controls a single drone with LiDAR scanning.
    Uses GPS-based alignment so all drones share the same world frame.
    """
    def __init__(self, drone_name: str, lidar_name: str = "Lidar1",
                 shared_client: Optional[airsim.MultirotorClient] = None,
                 lidar_offset_ned=(0.0, 0.0, -0.15),
                 lidar_rotation_rpy=(0.0, 0.0, 0.0)):
        self.name = drone_name
        self.lidar_name = lidar_name
        
        self._owns_client = (shared_client is None)
        self.client: Optional[airsim.MultirotorClient] = shared_client
        
        # Data directories
        self.data_dir = Path(DATA_DIR) / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.frame_idx = 0
        self.floor_z_enu: Optional[float] = None
        
        # GPS-based world alignment
        self._gps_origin: Optional[airsim.GeoPoint] = None
        self._calibrated = False
        
        # Assigned region (NED)
        self.assigned_region: Optional[Tuple[float, float, float, float]] = None
        
        # LiDAR sensor mounting (NED frame relative to drone body)
        self.lidar_offset_ned = np.array(lidar_offset_ned, dtype=np.float32)
        self.lidar_rotation_rpy = np.array(lidar_rotation_rpy, dtype=np.float32)
        
        # Pre-compute LiDAR rotation matrix (if sensor is rotated)
        if np.any(self.lidar_rotation_rpy != 0):
            self.lidar_rotation_matrix = R.from_euler('xyz', self.lidar_rotation_rpy).as_matrix()
        else:
            self.lidar_rotation_matrix = np.eye(3)

    def connect(self):
        """Establish dedicated AirSim connection."""
        if self._owns_client or self.client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self._owns_client = True
            print(f"[{self.name}] Connected")

    def disconnect(self):
        if self._owns_client and self.client is not None:
            self.client = None

    def _ensure_connected(self):
        if self.client is None:
            self.connect()

    def set_gps_origin(self, origin: airsim.GeoPoint):
        """Set shared GPS origin for world alignment."""
        self._gps_origin = origin
        self._calibrated = True
        print(f"[{self.name}] GPS origin set: lat={origin.latitude:.8f}, lon={origin.longitude:.8f}, alt={origin.altitude:.2f}")

    def calibrate_from_current_gps(self) -> airsim.GeoPoint:
        """Use this drone's current GPS as the origin."""
        self._ensure_connected()
        state = self.client.getMultirotorState(vehicle_name=self.name)
        self._gps_origin = state.gps_location
        self._calibrated = True
        print(f"[{self.name}] Set as GPS origin: lat={self._gps_origin.latitude:.8f}, "
              f"lon={self._gps_origin.longitude:.8f}, alt={self._gps_origin.altitude:.2f}")
        return self._gps_origin

    def _gps_to_enu(self, gps: airsim.GeoPoint) -> np.ndarray:
        """
        Convert GPS to ENU relative to origin.
        Uses flat-earth approximation (valid for < 10km distances).
        """
        if self._gps_origin is None:
            return np.zeros(3, dtype=np.float32)
        
        # Differences in degrees
        d_lat = gps.latitude - self._gps_origin.latitude
        d_lon = gps.longitude - self._gps_origin.longitude
        d_alt = gps.altitude - self._gps_origin.altitude
        
        # Convert to radians
        d_lat_rad = math.radians(d_lat)
        d_lon_rad = math.radians(d_lon)
        origin_lat_rad = math.radians(self._gps_origin.latitude)
        
        # Flat-earth approximation to meters
        # East-West distance accounts for latitude
        x_east = d_lon_rad * math.cos(origin_lat_rad) * R_EARTH
        y_north = d_lat_rad * R_EARTH
        z_up = d_alt
        
        return np.array([x_east, y_north, z_up], dtype=np.float32)

    def get_world_position_enu(self) -> np.ndarray:
        """Get current position in world ENU frame (GPS-aligned)."""
        self._ensure_connected()
        state = self.client.getMultirotorState(vehicle_name=self.name)
        
        if not self._calibrated:
            # Auto-calibrate on first call
            print(f"[{self.name}] Auto-calibrating GPS origin from current position")
            self._gps_origin = state.gps_location
            self._calibrated = True
        
        return self._gps_to_enu(state.gps_location)

    # ==================== Basic Flight ====================
    
    def takeoff(self, altitude: float = 5.0, speed: float = 2.0):
        self._ensure_connected()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)
        self.client.takeoffAsync(vehicle_name=self.name).join()
        self.client.moveToZAsync(-altitude, speed, vehicle_name=self.name).join()

    def land(self):
        self._ensure_connected()
        self.client.landAsync(vehicle_name=self.name).join()
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

    def get_position_ned(self) -> np.ndarray:
        self._ensure_connected()
        state = self.client.getMultirotorState(vehicle_name=self.name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def get_position_enu(self) -> np.ndarray:
        """Get local ENU position (for backward compatibility)."""
        ned = self.get_position_ned()
        return self.ned_to_enu_points(ned.reshape(1, 3)).flatten()

    # ==================== Coordinate Conversion ====================
    
    @staticmethod
    def ned_to_enu_points(pts_ned: np.ndarray) -> np.ndarray:
        """
        Convert NED points to ENU.
        NED: [X_ned, Y_ned, Z_ned] = [North, East, Down]
        ENU: [X_enu, Y_enu, Z_enu] = [East, North, Up]
        """
        if pts_ned.size == 0:
            return pts_ned
        return np.column_stack((
            pts_ned[:, 1],   # X_enu = Y_ned (East)
            pts_ned[:, 0],   # Y_enu = X_ned (North)
            -pts_ned[:, 2],  # Z_enu = -Z_ned (Up = -Down)
        ))

    @staticmethod
    def enu_to_ned(x_enu: float, y_enu: float, z_enu: float) -> Tuple[float, float, float]:
        return (y_enu, x_enu, -z_enu)

    # ==================== LiDAR Operations ====================

    def get_lidar_world_enu(self) -> np.ndarray:
        """
        COMPLETELY FIXED: Transforms LiDAR to GPS-aligned world frame.
        
        Strategy:
        1. Transform LiDAR points to local NED frame (body -> world NED)
        2. Get GPS positions for both origin and current drone
        3. Compute offset in meters using GPS differences
        4. Convert local NED to local ENU
        5. Add GPS-based offset to get world ENU
        """
        
        # STEP 1: Synchronized capture
        self.client.simPause(True)
        try:
            lidar_data = self.client.getLidarData(vehicle_name=self.name, lidar_name=self.lidar_name)
            state = self.client.getMultirotorState(vehicle_name=self.name)
        finally:
            self.client.simPause(False)

        if len(lidar_data.point_cloud) < 3:
            return None

        # STEP 2: Raw LiDAR points (sensor frame)
        pts_lidar_local = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        
        # STEP 3: Apply sensor rotation (if configured)
        pts_lidar_rotated = pts_lidar_local @ self.lidar_rotation_matrix.T
        
        # STEP 4: Add sensor offset (now in body frame)
        pts_body_ned = pts_lidar_rotated + self.lidar_offset_ned
        
        # STEP 5: Rotate to local world NED using drone orientation
        q = state.kinematics_estimated.orientation
        drone_rotation = R.from_quat([q.x_val, q.y_val, q.z_val, q.w_val])
        pts_local_ned_rotated = drone_rotation.apply(pts_body_ned)
        
        # STEP 6: Add drone position in local NED
        drone_pos_local_ned = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ], dtype=np.float32)
        
        pts_local_ned = pts_local_ned_rotated + drone_pos_local_ned
        
        # STEP 7: Convert local NED to local ENU
        pts_local_enu = self.ned_to_enu_points(pts_local_ned)
        
        # STEP 8: Add GPS-based world offset
        if self._calibrated and self._gps_origin is not None:
            # Get drone's GPS-based world position in ENU
            drone_gps_world_enu = self._gps_to_enu(state.gps_location)
            
            # Get drone's local position in ENU
            drone_local_enu = self.ned_to_enu_points(drone_pos_local_ned.reshape(1, 3)).flatten()
            
            # Compute the offset from local to world
            # world = local + (GPS_world - local_at_current_pos)
            offset = drone_gps_world_enu - drone_local_enu

            print(f"DEBUGG ... {drone_gps_world_enu}")
            print(f"DEBUGG ... {drone_local_enu}")
            print(f"DEBUGG...{offset}")

            # Apply to all points
            pts_world_enu = pts_local_enu + offset
        else:
            # No GPS calibration - use local ENU
            pts_world_enu = pts_local_enu
        
        return pts_world_enu
    
    def scan_and_save(self, colorize: bool = True) -> Optional[np.ndarray]:
        """Capture LiDAR in world frame and save."""
        pts_enu = self.get_lidar_world_enu()
        if pts_enu is None or pts_enu.shape[0] == 0:
            return None
        
        colors = None
        if colorize:
            colors = self._colorize_by_height(pts_enu)
        
        self._save_las(pts_enu, colors)
        self.frame_idx += 1
        return pts_enu

    def _colorize_by_height(self, pts_enu: np.ndarray) -> np.ndarray:
        z = pts_enu[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        denom = max(z_max - z_min, 1e-6)
        norm = (z - z_min) / denom
        colors = np.column_stack((norm, 0.5 * np.ones_like(norm), 1.0 - norm))
        
        if self.floor_z_enu is not None:
            floor_mask = np.abs(z - self.floor_z_enu) <= 0.2
            colors[floor_mask] = [0.0, 1.0, 0.0]
        
        return colors

    def _save_las(self, pts_enu: np.ndarray, colors: Optional[np.ndarray] = None):
        if pts_enu.size == 0:
            return
        
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision
        header.offsets = np.array([
            float(pts_enu[:, 0].min()),
            float(pts_enu[:, 1].min()),
            float(pts_enu[:, 2].min()),
        ])

        las = laspy.LasData(header)
        las.x = pts_enu[:, 0]
        las.y = pts_enu[:, 1]
        las.z = pts_enu[:, 2]

        if colors is not None and len(colors) == len(pts_enu):
            rgb16 = (np.clip(colors, 0.0, 1.0) * 65535.0).astype(np.uint16)
            las.red = rgb16[:, 0]
            las.green = rgb16[:, 1]
            las.blue = rgb16[:, 2]

        out_path = self.data_dir / f"lidar_{self.frame_idx:04d}.las"
        las.write(str(out_path))

    def estimate_floor(self, num_scans: int = 5, delay: float = 0.1) -> float:
        all_z = []
        for _ in range(num_scans):
            pts = self.get_lidar_world_enu()
            if pts is not None and pts.shape[0] > 0:
                all_z.append(pts[:, 2])
            time.sleep(delay)
        
        if not all_z:
            self.floor_z_enu = 0.0
            return 0.0
        
        z_all = np.concatenate(all_z)
        self.floor_z_enu = float(np.percentile(z_all, 5.0))
        return self.floor_z_enu

    def load_all_points(self) -> np.ndarray:
        """Load all saved LAS files."""
        files = sorted(self.data_dir.glob("*.las"))
        if not files:
            return np.empty((0, 3), dtype=np.float32)
        
        all_pts = []
        for f in files:
            try:
                las = laspy.read(f)
                pts = np.column_stack((
                    np.asarray(las.x, dtype=np.float32),
                    np.asarray(las.y, dtype=np.float32),
                    np.asarray(las.z, dtype=np.float32),
                ))
                all_pts.append(pts)
            except:
                pass
        
        return np.vstack(all_pts) if all_pts else np.empty((0, 3), dtype=np.float32)