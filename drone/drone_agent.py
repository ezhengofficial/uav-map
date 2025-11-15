import os
import numpy as np
import laspy
from setup_path import SetupPath

class DroneAgent:
    '''
    Class for one drone
    Contains functionally for one drone
    '''
    def __init__(self, client, drone_name, lidar_name="Lidar1"):
        self.client = client
        self.name = drone_name
        self.lidar_name = lidar_name
        self.lidar_data_dir = os.path.join( SetupPath.getLidarDataPath(), self.name)
        os.makedirs(self.lidar_data_dir, exist_ok=True)

    def takeoff(self, altitude=5):
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)
        self.client.takeoffAsync(vehicle_name=self.name).join()
        self.client.moveToZAsync(-altitude, 2, vehicle_name=self.name).join()

    def fly_waypoints(self, waypoints):
        for x, y, z in waypoints:
            self.client.moveToPositionAsync(x, y, z, 3, vehicle_name=self.name).join()

    def create_las(self, frame, points, colors=None,):
        """
        Create a LAS file from point cloud data
        :param points: numpy array of shape (N, 3) with x, y, z coordinates
        :param colors: numpy array of shape (N, 3) with RGB color data
        """
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])  # 1cm precision
        header.offsets = np.array([
                np.min(points[:,0]),
                np.min(points[:,1]),
                np.min(points[:,2])
            ])

        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        if colors is not None:
            rgb = np.clip(colors * 65535, 0, 65535).astype(np.uint16)
            las.red = rgb[:, 0]
            las.green = rgb[:, 1]
            las.blue = rgb[:, 2]

        # Write to file
        filename = os.path.join(self.lidar_data_dir, f"lidar_{frame:04d}.las")
        las.write(filename)
        print(f"[{self.name}] Saved LAS: {filename}")

    def get_lidar(self, frame):
        """
        Collect LiDAR data and save as both .npy and .las
        """
        lidar_data = self.client.getLidarData(
                vehicle_name=self.name,
                lidar_name=self.lidar_name
            )       
        if lidar_data.point_cloud:
            # Convert point cloud to numpy array
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            # Optional: colorize points by height (Z-value)
            colors = None
            if points.shape[0] > 0:
                colors = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
                colors = np.column_stack((colors, 0.5 * np.ones_like(colors), 1 - colors))  # R -> B color map

            # Save as .npy
            np.save(os.path.join(self.lidar_data_dir, f"lidar_{frame:04d}.npy"), points)
            print(f"[{self.name}] Saved NPY: lidar_{frame:04d}.npy")   

            # Save as .las (LiDAR point cloud in LAS format)
            self.create_las(frame, points, colors)
