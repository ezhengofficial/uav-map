import json
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import airsim

from drone_agent import DroneAgent
from exploration.explorer import Explorer, ExplorationConfig


class DroneManager:
    """
    Manages fleet of drones with support for parallel autonomous exploration.
    
    Key features:
    - Each drone gets its own AirSim client for thread-safe parallel operation
    - Automatic map partitioning for multi-drone exploration
    """
    
    def __init__(self, settings_file: Optional[str] = None):
        # Single shared client for manager-level operations
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        self.drones: Dict[str, DroneAgent] = {}
        self._parse_settings(settings_file)
        
        # LiDAR logging state (thread-safe)
        self._lidar_lock = threading.Lock()
        self._lidar_logging = False
        self._lidar_thread: Optional[threading.Thread] = None
        self._lidar_stop_event = threading.Event()
        
        # Exploration
        self._explorers: Dict[str, Explorer] = {}
        self._explore_threads: Dict[str, threading.Thread] = {}
        
        # Global exploration bounds (can be overridden)
        self.world_bounds_ned: Tuple[float, float, float, float] = (-150, 150, -150, 150)

    def _parse_settings(self, settings_file: Optional[str]):
        """Parse AirSim settings.json to discover drones."""
        if settings_file is None:
            # Cross-platform default paths
            candidates = [
                Path.home() / "Documents" / "AirSim" / "settings.json",
                Path.home() / ".airsim" / "settings.json",
                Path("/etc/airsim/settings.json"),
            ]
            for p in candidates:
                if p.exists():
                    settings_file = str(p)
                    break
        
        if settings_file is None or not os.path.isfile(settings_file):
            raise FileNotFoundError(
                f"AirSim settings.json not found. Tried: {candidates}"
            )
        
        with open(settings_file) as f:
            settings = json.load(f)
        
        vehicles = settings.get("Vehicles", {})
        if not vehicles:
            raise ValueError("No vehicles found in settings.json")
        
        for name, vehicle_cfg in vehicles.items():
            # Find LiDAR sensor
            lidar_name = "Lidar1"  # default fallback
            sensors = vehicle_cfg.get("Sensors", {})
            for sensor_name, sensor_cfg in sensors.items():
                if sensor_cfg.get("SensorType") == 6:  # LiDAR
                    lidar_name = sensor_name
                    break
            
            # Create DroneAgent WITHOUT shared client (will connect on its own)
            self.drones[name] = DroneAgent(name, lidar_name, shared_client=None)
            print(f"[Manager] Registered drone: {name} (LiDAR: {lidar_name})")
        
        print(f"[Manager] Total drones: {len(self.drones)}")

    def set_world_bounds(self, x_min: float, x_max: float, 
                         y_min: float, y_max: float):
        """Set global exploration bounds in NED coordinates."""
        self.world_bounds_ned = (x_min, x_max, y_min, y_max)
        print(f"[Manager] World bounds NED: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")

    # ==================== Map Partitioning ====================
    
    def partition_map(self, num_partitions: Optional[int] = None,
                      strategy: str = "vertical") -> List[Tuple[float, float, float, float]]:
        """
        Partition the world into regions for multi-drone exploration.
        
        Args:
            num_partitions: Number of partitions (default: number of drones)
            strategy: "vertical" (split on X), "horizontal" (split on Y), 
                     "grid" (2D grid)
        
        Returns:
            List of (x_min, x_max, y_min, y_max) tuples in NED
        """
        if num_partitions is None:
            num_partitions = len(self.drones)
        
        x_min, x_max, y_min, y_max = self.world_bounds_ned
        
        if strategy == "vertical":
            # Split along X axis (North-South strips)
            regions = []
            x_step = (x_max - x_min) / num_partitions
            for i in range(num_partitions):
                rx_min = x_min + i * x_step
                rx_max = x_min + (i + 1) * x_step
                regions.append((rx_min, rx_max, y_min, y_max))
            return regions
            
        elif strategy == "horizontal":
            # Split along Y axis (East-West strips)
            regions = []
            y_step = (y_max - y_min) / num_partitions
            for i in range(num_partitions):
                ry_min = y_min + i * y_step
                ry_max = y_min + (i + 1) * y_step
                regions.append((x_min, x_max, ry_min, ry_max))
            return regions
            
        elif strategy == "grid":
            # 2D grid partitioning
            import math
            cols = int(math.ceil(math.sqrt(num_partitions)))
            rows = int(math.ceil(num_partitions / cols))
            
            x_step = (x_max - x_min) / cols
            y_step = (y_max - y_min) / rows
            
            regions = []
            for r in range(rows):
                for c in range(cols):
                    if len(regions) >= num_partitions:
                        break
                    rx_min = x_min + c * x_step
                    rx_max = x_min + (c + 1) * x_step
                    ry_min = y_min + r * y_step
                    ry_max = y_min + (r + 1) * y_step
                    regions.append((rx_min, rx_max, ry_min, ry_max))
            return regions
        
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

    def assign_regions(self, strategy: str = "vertical"):
        """
        Automatically partition map and assign regions to drones.
        """
        regions = self.partition_map(len(self.drones), strategy)
        
        for drone, region in zip(self.drones.values(), regions):
            drone.assigned_region = region
            x_min, x_max, y_min, y_max = region
            print(f"[Manager] {drone.name} assigned region: "
                  f"X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")

    # ==================== Basic Fleet Control ====================
    
    def takeoff_all(self, altitude: float = 5.0):
        """Takeoff all drones in parallel."""
        print("[Manager] Taking off all drones...")
        threads = []
        for drone in self.drones.values():
            def do_takeoff(d=drone):
                d.connect()  # Each drone gets its own client
                d.takeoff(altitude)
            t = threading.Thread(target=do_takeoff)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("[Manager] All drones airborne")

    def land_all(self):
        """Land all drones."""
        print("[Manager] Landing all drones...")
        threads = []
        for drone in self.drones.values():
            t = threading.Thread(target=drone.land)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("[Manager] All drones landed")

    def stop_all(self):
        """Hover all drones in place."""
        for drone in self.drones.values():
            drone._ensure_connected()
            drone.client.hoverAsync(vehicle_name=drone.name).join()
            print(f"[{drone.name}] Hovering")

    # ==================== LiDAR Logging ====================
    
    def start_lidar_logging(self, interval: float = 1.0):
        """Start background LiDAR capture for all drones."""
        with self._lidar_lock:
            if self._lidar_logging:
                print("[Manager] LiDAR logging already running")
                return
            
            self._lidar_logging = True
            self._lidar_stop_event.clear()
        
        # Ensure all drones have connections
        for drone in self.drones.values():
            drone.connect()
        
        def log_loop():
            while not self._lidar_stop_event.is_set():
                for drone in self.drones.values():
                    drone.scan_and_save()
                time.sleep(interval)
        
        self._lidar_thread = threading.Thread(target=log_loop, daemon=True)
        self._lidar_thread.start()
        print(f"[Manager] Started LiDAR logging (interval={interval}s)")

    def stop_lidar_logging(self):
        """Stop background LiDAR capture."""
        with self._lidar_lock:
            if not self._lidar_logging:
                print("[Manager] LiDAR logging not running")
                return
            self._lidar_logging = False
        
        self._lidar_stop_event.set()
        if self._lidar_thread:
            self._lidar_thread.join(timeout=3.0)
            self._lidar_thread = None
        print("[Manager] Stopped LiDAR logging")

    # ==================== Parallel Autonomous Exploration ====================
    
    def start_exploration(self, 
                          drone_names: Optional[List[str]] = None,
                          config: Optional[ExplorationConfig] = None,
                          partition_strategy: str = "vertical"):
        """
        Start autonomous exploration for specified drones IN PARALLEL.
        
        Args:
            drone_names: List of drone names, or None for all drones
            config: Exploration configuration
            partition_strategy: How to divide the map ("vertical", "horizontal", "grid")
        """
        if drone_names is None:
            drone_names = list(self.drones.keys())
        
        # Filter to valid drones
        targets = [name for name in drone_names if name in self.drones]
        
        if not targets:
            print("[Manager] No valid drones to start exploration")
            return
        
        # Auto-partition map among participating drones
        regions = self.partition_map(len(targets), partition_strategy)
        
        print(f"[Manager] Starting parallel exploration with {len(targets)} drones")
        print(f"[Manager] Partition strategy: {partition_strategy}")
        
        for name, region in zip(targets, regions):
            if name in self._explore_threads and self._explore_threads[name].is_alive():
                print(f"[Manager] {name} already exploring")
                continue
            
            drone = self.drones[name]
            drone.assigned_region = region
            
            explorer = Explorer(drone, config)
            explorer.set_region(*region)
            self._explorers[name] = explorer
            
            # Start exploration in separate thread
            thread = threading.Thread(target=explorer.run, daemon=True, name=f"explore-{name}")
            thread.start()
            self._explore_threads[name] = thread
            
            x_min, x_max, y_min, y_max = region
            print(f"[Manager] {name} exploring region: "
                  f"X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")

    def get_exploration_status(self) -> Dict[str, bool]:
        """Check which drones are still exploring."""
        return {
            name: thread.is_alive() 
            for name, thread in self._explore_threads.items()
        }

    def wait_for_exploration(self, timeout: Optional[float] = None):
        """Wait for all exploration threads to complete."""
        for name, thread in self._explore_threads.items():
            if thread.is_alive():
                thread.join(timeout=timeout)
                print(f"[Manager] {name} exploration finished")

    # ==================== Cleanup ====================
    
    def shutdown(self):
        """Clean shutdown of all resources."""
        print("[Manager] Shutting down...")
        self.stop_lidar_logging()
        
        # Signal explorers to stop (they handle their own landing)
        # Wait briefly for them to finish current iteration
        for name, thread in self._explore_threads.items():
            if thread.is_alive():
                print(f"[Manager] Waiting for {name} to finish...")
                thread.join(timeout=10.0)
        
        # Land any drones that might not have landed
        self.land_all()
        
        # Disconnect all drones
        for drone in self.drones.values():
            drone.disconnect()
        
        print("[Manager] Shutdown complete")