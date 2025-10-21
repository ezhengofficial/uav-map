import glob, numpy as np, open3d as o3d
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "data" / "logs"
files = sorted(glob.glob(str(LOGS / "lidar_*.npy")))
if not files:
    raise SystemExit(f"No lidar_*.npy found in {LOGS}")

clouds = []
for f in files:
    pts = np.load(f)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel_size=0.10)
    clouds.append(pcd)

merged = o3d.geometry.PointCloud()
for p in clouds:
    merged += p

merged, _ = merged.remove_radius_outlier(nb_points=8, radius=0.3)
out = Path(__file__).resolve().parents[1] / "data" / "merged.ply"
o3d.io.write_point_cloud(str(out), merged)
print(f"Wrote {out}")
o3d.visualization.draw_geometries([merged])
