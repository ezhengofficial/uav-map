import airsim, time, os, numpy as np
from pathlib import Path
import laspy


SAVE = Path(__file__).resolve().parents[1] / "data" / "logs"
SAVE.mkdir(parents=True, exist_ok=True)

c = airsim.MultirotorClient()
c.confirmConnection()
c.enableApiControl(True)
c.armDisarm(True)

def create_las(frame, points, colors=None,):
    """
    Create a LAS file from point cloud data
    :param points: numpy array of shape (N, 3) with x, y, z coordinates
    :param colors: numpy array of shape (N, 3) with RGB color data
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])  # 1cm precision
    header.offsets = np.array([np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])])

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
    las.write(str(SAVE / f"lidar_{frame:04d}.las"))
    print(f"Saved: lidar_{frame:04d}.las")

def log_lidar(frame):
    """
    Collect and save LiDAR data (both .npy and .las)
    """
    lidar_data = c.getLidarData(vehicle_name="Drone1", lidar_name="Lidar1")
    
    if lidar_data.point_cloud:
        # Convert point cloud to numpy array
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Optional: colorize points by height (Z-value)
        colors = None
        if points.shape[0] > 0:
            colors = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
            colors = np.column_stack((colors, 0.5 * np.ones_like(colors), 1 - colors))  # R -> B color map

        # Save as .npy
        np.save(SAVE / f"lidar_{frame:04d}.npy", points)
        
        # Save as .las (LiDAR point cloud in LAS format)
        create_las(frame, points, colors)

print("Taking off...")
c.takeoffAsync().join()
c.moveToZAsync(-5, 2).join()   # hover about 5 m above ground


# simple square in world frame (meters)
wps = [(10,0,-5), (10,10,-5), (0,10,-5), (0,0,-5)]
idx = 0

# def log_lidar(frame):
#     lidar = c.getLidarData(vehicle_name="Drone1", lidar_name="Lidar1")
#     if lidar.point_cloud:
#         pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1,3)
#         np.save(SAVE / f"lidar_{frame:04d}.npy", pts)

for step in range(40):
    if step % 10 == 0:
        x,y,z = wps[idx]; idx = (idx+1) % len(wps)
        c.moveToPositionAsync(x,y,z, 3)
    log_lidar(step)
    time.sleep(1)

c.landAsync().join()
c.armDisarm(False); c.enableApiControl(False)
print(f"Done. LiDAR frames saved to {SAVE}")
