import airsim, time, os, numpy as np
from pathlib import Path

SAVE = Path(__file__).resolve().parents[1] / "data" / "logs"
SAVE.mkdir(parents=True, exist_ok=True)

c = airsim.MultirotorClient()
c.confirmConnection()
c.enableApiControl(True)
c.armDisarm(True)

print("Taking off...")
c.takeoffAsync().join()
c.moveToZAsync(-5, 2).join()   # hover about 5 m above ground


# simple square in world frame (meters)
wps = [(10,0,-5), (10,10,-5), (0,10,-5), (0,0,-5)]
idx = 0

def log_lidar(frame):
    lidar = c.getLidarData(vehicle_name="Drone1", lidar_name="Lidar1")
    if lidar.point_cloud:
        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1,3)
        np.save(SAVE / f"lidar_{frame:04d}.npy", pts)

for step in range(40):
    if step % 10 == 0:
        x,y,z = wps[idx]; idx = (idx+1) % len(wps)
        c.moveToPositionAsync(x,y,z, 3)
    log_lidar(step)
    time.sleep(1)

c.landAsync().join()
c.armDisarm(False); c.enableApiControl(False)
print(f"Done. LiDAR frames saved to {SAVE}")
