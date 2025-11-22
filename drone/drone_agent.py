import os
import time
import numpy as np
import laspy
import airsim
from pathlib import Path
from typing import Optional, Tuple
from files_path import DATA_DIR


class DroneAgent:
    """
    Controls a single drone with LiDAR scanning and exploration support.
    
    IMPORTANT: Each DroneAgent creates its own AirSim client connection
    to enable thread-safe parallel operation.
    """
    def __init__(self, drone_name: str, lidar_name: str = "Lidar1",
                 shared_client: Optional[airsim.MultirotorClient] = None):
        """
        Args:
            drone_name: Name from settings.json
            lidar_name: LiDAR sensor name
            shared_client: Optional shared client for single-threaded use.
                          If None, creates a dedicated client (required for multi-threading).
        """
        self.name = drone_name
        self.lidar_name = lidar_name
        
        # Thread-safe: each agent can have its own client
        self._owns_client = (shared_client is None)
        self.client: Optional[airsim.MultirotorClient] = shared_client
        
        # Data directories
        self.data_dir = Path(DATA_DIR) / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Exploration state
        self.frame_idx = 0
        self.floor_z_enu: Optional[float] = None
        
        # Assigned region for partitioned exploration (set by manager)
        self.assigned_region: Optional[Tuple[float, float, float, float]] = None  # (x_min, x_max, y_min, y_max) in NED

    def connect(self):
        """
        Establish dedicated AirSim connection for this drone.
        Call this before any flight operations if using multi-threaded exploration.
        """
        if self._owns_client or self.client is None:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self._owns_client = True
            print(f"[{self.name}] Connected to AirSim (dedicated client)")

    def disconnect(self):
        """Clean up dedicated client connection."""
        if self._owns_client and self.client is not None:
            # AirSim client doesn't have explicit disconnect, but we can clear it
            self.client = None
            print(f"[{self.name}] Disconnected")

    def _ensure_connected(self):
        """Ensure we have a valid client connection."""
        if self.client is None:
            self.connect()
        
    # ==================== Basic Flight ====================
    
    def takeoff(self, altitude: float = 5.0, speed: float = 2.0):
        """Arm, takeoff, and climb to specified altitude (positive meters up)."""
        self._ensure_connected()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)
        self.client.takeoffAsync(vehicle_name=self.name).join()
        # NED: negative Z is up
        self.client.moveToZAsync(-altitude, speed, vehicle_name=self.name).join()
        print(f"[{self.name}] Takeoff complete at {altitude}m")

    def land(self):
        """Land and disarm the drone."""
        self._ensure_connected()
        self.client.landAsync(vehicle_name=self.name).join()
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
        print(f"[{self.name}] Landed and disarmed")

    def move_to_ned(self, x: float, y: float, z: float, speed: float = 3.0,
                    timeout: float = 30.0):
        """Move to NED position."""
        self._ensure_connected()
        self.client.moveToPositionAsync(
            x, y, z, speed, timeout_sec=timeout, vehicle_name=self.name
        ).join()

    def move_to_enu(self, x_enu: float, y_enu: float, z_enu: float, 
                    speed: float = 3.0, timeout: float = 30.0):
        """Move to ENU position (converts to NED internally)."""
        x_ned, y_ned, z_ned = self.enu_to_ned(x_enu, y_enu, z_enu)
        self.move_to_ned(x_ned, y_ned, z_ned, speed, timeout)

    def get_position_ned(self) -> np.ndarray:
        """Get current position in NED coordinates."""
        self._ensure_connected()
        state = self.client.getMultirotorState(vehicle_name=self.name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def get_position_enu(self) -> np.ndarray:
        """Get current position in ENU coordinates."""
        ned = self.get_position_ned()
        return self.ned_to_enu_points(ned.reshape(1, 3)).flatten()

    # ==================== Coordinate Conversion ====================
    
    @staticmethod
    def ned_to_enu_points(pts_ned: np.ndarray) -> np.ndarray:
        """Convert NED points (N,3) to ENU."""
        if pts_ned.size == 0:
            return pts_ned
        return np.column_stack((
            pts_ned[:, 1],   # X_enu = Y_ned (East)
            pts_ned[:, 0],   # Y_enu = X_ned (North)
            -pts_ned[:, 2],  # Z_enu = -Z_ned (Up)
        ))

    @staticmethod
    def enu_to_ned(x_enu: float, y_enu: float, z_enu: float) -> Tuple[float, float, float]:
        """Convert single ENU point to NED."""
        return (y_enu, x_enu, -z_enu)  # X_ned=N=y_enu, Y_ned=E=x_enu, Z_ned=-z_enu

    # ==================== LiDAR Operations ====================

    def get_lidar_enu(self) -> Optional[np.ndarray]:
        """
        Capture LiDAR frame and return points in ENU coordinates.
        Returns None if no points captured.
        """
        self._ensure_connected()
        lidar_data = self.client.getLidarData(
            vehicle_name=self.name,
            lidar_name=self.lidar_name
        )
        if not lidar_data.point_cloud:
            return None
            
        pts_ned = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        if pts_ned.shape[0] == 0:
            return None
            
        return self.ned_to_enu_points(pts_ned)

    def scan_and_save(self, colorize: bool = True) -> Optional[np.ndarray]:
        """
        Capture LiDAR frame, save to LAS file, return ENU points.
        """
        pts_enu = self.get_lidar_enu()
        if pts_enu is None or pts_enu.shape[0] == 0:
            print(f"[{self.name}] Frame {self.frame_idx}: empty scan")
            return None
        
        colors = None
        if colorize:
            colors = self._colorize_by_height(pts_enu)
        
        self._save_las(pts_enu, colors)
        self.frame_idx += 1
        return pts_enu

    def _colorize_by_height(self, pts_enu: np.ndarray) -> np.ndarray:
        """Generate height-based colors with floor highlighting."""
        z = pts_enu[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        
        # Avoid division by zero
        denom = max(z_max - z_min, 1e-6)
        norm = (z - z_min) / denom
        
        # Red-to-blue gradient
        colors = np.column_stack((norm, 0.5 * np.ones_like(norm), 1.0 - norm))
        
        # Highlight floor in green if detected
        if self.floor_z_enu is not None:
            floor_mask = np.abs(z - self.floor_z_enu) <= 0.1
            colors[floor_mask] = [0.0, 1.0, 0.0]
        
        return colors

    def _save_las(self, pts_enu: np.ndarray, colors: Optional[np.ndarray] = None):
        """Write points to LAS file."""
        if pts_enu.size == 0:
            return
            
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
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
            rgb16 = (np.clip(colors, 0.0, 1.0) * 65535).astype(np.uint16)
            las.red = rgb16[:, 0]
            las.green = rgb16[:, 1]
            las.blue = rgb16[:, 2]

        out_path = self.data_dir / f"lidar_{self.frame_idx:04d}.las"
        las.write(str(out_path))
        print(f"[{self.name}] Saved {out_path.name} ({len(pts_enu):,} pts)")

    def estimate_floor(self, num_scans: int = 5, delay: float = 0.1) -> float:
        """Estimate floor Z in ENU from several LiDAR scans."""
        all_z = []
        for _ in range(num_scans):
            pts = self.get_lidar_enu()
            if pts is not None and pts.shape[0] > 0:
                all_z.append(pts[:, 2])
            time.sleep(delay)
        
        if not all_z:
            print(f"[{self.name}] Floor estimation failed, defaulting to 0.0")
            self.floor_z_enu = 0.0
            return 0.0
        
        z_all = np.concatenate(all_z)
        self.floor_z_enu = float(np.percentile(z_all, 5.0))
        print(f"[{self.name}] Floor Z (ENU) ≈ {self.floor_z_enu:.3f}m")
        return self.floor_z_enu

    def load_all_points(self) -> np.ndarray:
        """Load all saved LAS files and return combined ENU points."""
        files = sorted(self.data_dir.glob("*.las"))
        if not files:
            return np.empty((0, 3), dtype=np.float32)
        
        all_pts = []
        for f in files:
            las = laspy.read(f)
            pts = np.column_stack((
                np.asarray(las.x, dtype=np.float32),
                np.asarray(las.y, dtype=np.float32),
                np.asarray(las.z, dtype=np.float32),
            ))
            all_pts.append(pts)
        
        return np.vstack(all_pts) if all_pts else np.empty((0, 3), dtype=np.float32)