"""
DroneManager with GPS calibration for world alignment and ENU partitioning.
"""
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
    Manages drone fleet with GPS-aligned world frame.
    All drones share the same coordinate system via GPS calibration.
    """
    
    def __init__(self, settings_file: Optional[str] = None):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        self.drones: Dict[str, DroneAgent] = {}
        self._parse_settings(settings_file)
        
        # GPS origin for world alignment
        self._gps_origin: Optional[airsim.GeoPoint] = None
        
        # Thread management
        self._lidar_lock = threading.Lock()
        self._lidar_logging = False
        self._lidar_thread: Optional[threading.Thread] = None
        self._lidar_stop_event = threading.Event()
        
        self._explorers: Dict[str, Explorer] = {}
        self._explore_threads: Dict[str, threading.Thread] = {}
        
        # World bounds in ENU (after GPS calibration)
        self.world_bounds_enu: Tuple[float, float, float, float] = (-150, 150, -150, 150)

    def _parse_settings(self, settings_file: Optional[str]):
        if settings_file is None:
            candidates = [
                Path.home() / "Documents" / "AirSim" / "settings.json",
                Path.home() / ".airsim" / "settings.json",
            ]
            for p in candidates:
                if p.exists():
                    settings_file = str(p)
                    break
        
        if settings_file is None or not os.path.isfile(settings_file):
            raise FileNotFoundError(f"settings.json not found")
        
        with open(settings_file) as f:
            settings = json.load(f)
        
        vehicles = settings.get("Vehicles", {})
        if not vehicles:
            raise ValueError("No vehicles in settings.json")
        
        for name, cfg in vehicles.items():
            lidar_name = "Lidar1"
            for sname, scfg in cfg.get("Sensors", {}).items():
                if scfg.get("SensorType") == 6:
                    lidar_name = sname
                    break
            
            self.drones[name] = DroneAgent(name, lidar_name, shared_client=None)
            print(f"[Manager] Registered: {name}")
        
        print(f"[Manager] {len(self.drones)} drones total")

    def calibrate_gps_origin(self, reference_drone: Optional[str] = None):
        """
        Calibrate world origin using GPS from reference drone.
        All drones will use this origin for alignment.
        """
        if reference_drone is None:
            reference_drone = list(self.drones.keys())[0]
        
        drone = self.drones[reference_drone]
        drone.connect()
        
        self._gps_origin = drone.calibrate_from_current_gps()
        
        # Share origin with all drones
        for d in self.drones.values():
            d.set_gps_origin(self._gps_origin)
        
        print(f"[Manager] GPS origin set from {reference_drone}")
        return self._gps_origin

    def set_world_bounds_enu(self, x_min: float, x_max: float, 
                             y_min: float, y_max: float):
        """Set world bounds in ENU coordinates."""
        self.world_bounds_enu = (x_min, x_max, y_min, y_max)
        print(f"[Manager] World ENU: X=[{x_min},{x_max}] Y=[{y_min},{y_max}]")

    def partition_map_enu(self, num_partitions: Optional[int] = None,
                          strategy: str = "quadrant") -> List[Tuple[float, float, float, float]]:
        """
        Partition world into ENU regions.
        Returns list of (x_min, x_max, y_min, y_max) in ENU.
        """
        if num_partitions is None:
            num_partitions = len(self.drones)
        
        x_min, x_max, y_min, y_max = self.world_bounds_enu
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        margin = 10.0  # Overlap
        
        if strategy == "quadrant":
            if num_partitions == 1:
                return [(x_min, x_max, y_min, y_max)]
            elif num_partitions == 2:
                return [
                    (x_min, cx + margin, y_min, y_max),
                    (cx - margin, x_max, y_min, y_max),
                ]
            elif num_partitions <= 4:
                regions = [
                    (x_min, cx + margin, y_min, cy + margin),
                    (cx - margin, x_max, y_min, cy + margin),
                    (x_min, cx + margin, cy - margin, y_max),
                    (cx - margin, x_max, cy - margin, y_max),
                ]
                return regions[:num_partitions]
            else:
                strategy = "grid"
        
        if strategy == "vertical":
            regions = []
            step = (x_max - x_min) / num_partitions
            for i in range(num_partitions):
                rx_min = x_min + i * step - (margin if i > 0 else 0)
                rx_max = x_min + (i + 1) * step + (margin if i < num_partitions-1 else 0)
                regions.append((rx_min, rx_max, y_min, y_max))
            return regions
        
        if strategy == "horizontal":
            regions = []
            step = (y_max - y_min) / num_partitions
            for i in range(num_partitions):
                ry_min = y_min + i * step - (margin if i > 0 else 0)
                ry_max = y_min + (i + 1) * step + (margin if i < num_partitions-1 else 0)
                regions.append((x_min, x_max, ry_min, ry_max))
            return regions
        
        if strategy == "grid":
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
                    rx_min = x_min + c * x_step - (margin if c > 0 else 0)
                    rx_max = x_min + (c + 1) * x_step + (margin if c < cols-1 else 0)
                    ry_min = y_min + r * y_step - (margin if r > 0 else 0)
                    ry_max = y_min + (r + 1) * y_step + (margin if r < rows-1 else 0)
                    regions.append((rx_min, rx_max, ry_min, ry_max))
            return regions
        
        raise ValueError(f"Unknown strategy: {strategy}")

    # ==================== Flight Control ====================
    
    def takeoff_all(self, altitude: float = 5.0):
        print("[Manager] Taking off...")
        threads = []
        for drone in self.drones.values():
            def do_takeoff(d=drone):
                d.connect()
                d.takeoff(altitude)
            t = threading.Thread(target=do_takeoff)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("[Manager] All airborne")

    def land_all(self):
        print("[Manager] Landing...")
        threads = []
        for drone in self.drones.values():
            t = threading.Thread(target=drone.land)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def stop_all(self):
        for drone in self.drones.values():
            drone._ensure_connected()
            drone.client.hoverAsync(vehicle_name=drone.name).join()

    # ==================== Exploration ====================
    
    def start_exploration(self, 
                          drone_names: Optional[List[str]] = None,
                          config: Optional[ExplorationConfig] = None,
                          partition_strategy: str = "quadrant"):
        """Start parallel exploration with GPS-aligned world frame."""
        
        if drone_names is None:
            drone_names = list(self.drones.keys())
        
        targets = [n for n in drone_names if n in self.drones]
        if not targets:
            print("[Manager] No valid drones")
            return
        
        # Calibrate GPS if not done
        if self._gps_origin is None:
            self.calibrate_gps_origin(targets[0])
        
        # Partition in ENU
        regions = self.partition_map_enu(len(targets), partition_strategy)
        
        print(f"[Manager] Starting {len(targets)} drones with {partition_strategy} partitioning")
        
        for name, region in zip(targets, regions):
            if name in self._explore_threads and self._explore_threads[name].is_alive():
                continue
            
            drone = self.drones[name]
            explorer = Explorer(drone, config)
            explorer.set_region_enu(*region)  # Set ENU region directly
            self._explorers[name] = explorer
            
            thread = threading.Thread(target=explorer.run, daemon=True, name=f"explore-{name}")
            thread.start()
            self._explore_threads[name] = thread
            
            print(f"[Manager] {name} -> ENU region X=[{region[0]:.0f},{region[1]:.0f}] Y=[{region[2]:.0f},{region[3]:.0f}]")

    def get_exploration_status(self) -> Dict[str, bool]:
        return {name: t.is_alive() for name, t in self._explore_threads.items()}

    def wait_for_exploration(self, timeout: Optional[float] = None):
        for name, thread in self._explore_threads.items():
            if thread.is_alive():
                thread.join(timeout=timeout)
                print(f"[Manager] {name} finished")

    def shutdown(self):
        print("[Manager] Shutting down...")
        
        for name, thread in self._explore_threads.items():
            if thread.is_alive():
                thread.join(timeout=10.0)
        
        self.land_all()
        
        for drone in self.drones.values():
            drone.disconnect()
        
        print("[Manager] Done")