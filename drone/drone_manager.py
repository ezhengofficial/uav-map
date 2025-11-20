import json
import os
import airsim

import time
import threading

from drone_agent import DroneAgent

from ..config import DATA_DIR

class DroneManager:
    '''
    Class to Manage all the drones.
    TODO - Incorporate automonous drone management 
    '''
    def __init__(self, settings_file=None):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self._parse_settings_json(settings_file)

        self._lidar_logging = False
        self._lidar_thread = None

    def _parse_settings_json(self, settings_file):
        if settings_file is None:
            settings_file = os.path.expanduser("~/Documents/AirSim/settings.json")
        # Assert that the file exists
        assert os.path.isfile(settings_file), f"Settings file not found: {settings_file}"
        # Assert that the file is non-empty
        assert os.path.getsize(settings_file) > 0, f"Settings file is empty: {settings_file}"
        # Parse settings.json
        with open(settings_file) as f:
            settings = json.load(f)
        
        self.drones = {}
        for name, vehicle in settings.get("Vehicles", {}).items():
            # Default lidar name if none found
            lidar_name = None

            # Look inside Sensors for a LiDAR
            sensors = vehicle.get("Sensors", {})
            for sensor_name, sensor_cfg in sensors.items():
                if sensor_cfg.get("SensorType") == 6:  # 6 = LiDAR in AirSim
                    lidar_name = sensor_name
                    # break

            if lidar_name is None:
                print(f"[WARNING] No LiDAR found for {name}, skipping lidar assignment.")
                lidar_name = "Lidar1"  # fallback if needed

            # Pass lidar_name into DroneAgent
            self.drones[name] = DroneAgent(self.client, name, lidar_name=lidar_name)
            print(f"Initialized drones: {list(self.drones.keys())}")
        
    def takeoff_all(self, altitude = 5):
        print("Starting Takeoff")
        for drone in self.drones.values():
            drone.takeoff(altitude)
    
    def land_all(self):
        for drone in self.drones.values():
            self.client.landAsync(vehicle_name=drone.name).join()
            self.client.armDisarm(False, drone.name)
            self.client.enableApiControl(False, drone.name)
            print(f"[{drone.name}] Landed and disarmed.")

    def stop_all(self):
        for drone in self.drones.values():
            self.client.hoverAsync(vehicle_name=drone.name).join()
            print(f"[{drone.name}] Hovering (stopped).")

    def start_lidar_logging(self, interval=1.0):
        if self._lidar_logging:
            print("LiDAR logging already running")
            return
        self._lidar_logging = True

        def _log_loop():
            frame = 0
            while self._lidar_logging:
                for drone in self.drones.values():
                    drone.get_lidar(frame)
                frame += 1
                time.sleep(interval)

        self._lidar_thread = threading.Thread(target=_log_loop, daemon=True)
        self._lidar_thread.start()
        print("Started LiDAR logging for all drones.")

    def stop_lidar_logging(self):
        """Stop capturing LiDAR data."""
        if not self._lidar_logging:
            print("LiDAR logging is not running.")
            return
        self._lidar_logging = False
        if self._lidar_thread:
            self._lidar_thread.join(timeout=2)
        print("Stopped LiDAR logging for all drones.")
        