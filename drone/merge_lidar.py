"""
Merge LiDAR data from multiple drones into a single aligned point cloud.

Since all drones now save data in a shared world ENU frame (via GPS calibration),
merging is simply concatenating all points.

Usage:
    python merge_lidar.py                    # Merge all drones
    python merge_lidar.py --output merged.las
    python merge_lidar.py --drones Drone1 Drone2
"""
import argparse
from pathlib import Path
import numpy as np
import laspy
from datetime import datetime

from files_path import DATA_DIR


def load_drone_points(drone_dir: Path) -> tuple:
    """Load all LAS files from a drone directory."""
    files = sorted(drone_dir.glob("*.las"))
    
    if not files:
        return np.empty((0, 3)), np.empty((0, 3))
    
    all_pts = []
    all_colors = []
    
    for f in files:
        try:
            las = laspy.read(f)
            pts = np.column_stack((
                np.asarray(las.x, dtype=np.float32),
                np.asarray(las.y, dtype=np.float32),
                np.asarray(las.z, dtype=np.float32),
            ))
            all_pts.append(pts)
            
            # Try to get colors
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.column_stack((
                    np.asarray(las.red, dtype=np.float32) / 65535.0,
                    np.asarray(las.green, dtype=np.float32) / 65535.0,
                    np.asarray(las.blue, dtype=np.float32) / 65535.0,
                ))
                all_colors.append(colors)
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
    
    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3))
    
    pts = np.vstack(all_pts)
    colors = np.vstack(all_colors) if all_colors else np.empty((0, 3))
    
    return pts, colors


def find_drone_dirs(data_dir: Path, drone_names: list = None) -> list:
    """Find drone directories."""
    if drone_names:
        return [data_dir / name for name in drone_names if (data_dir / name).exists()]
    
    drones = []
    for d in data_dir.iterdir():
        if d.is_dir() and d.name.startswith("Drone"):
            if list(d.glob("*.las")):
                drones.append(d)
    return sorted(drones)


def downsample_points(pts: np.ndarray, colors: np.ndarray, 
                      voxel_size: float) -> tuple:
    """Voxel-based downsampling to reduce point count."""
    if pts.size == 0 or voxel_size <= 0:
        return pts, colors
    
    # Quantize to voxel grid
    voxel_idx = np.floor(pts / voxel_size).astype(np.int32)
    
    # Get unique voxels
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    
    pts_down = pts[unique_idx]
    colors_down = colors[unique_idx] if colors.size > 0 else colors
    
    return pts_down, colors_down


def save_las(output_path: Path, points: np.ndarray, colors: np.ndarray = None):
    """Save points to LAS file."""
    if points.size == 0:
        print("[WARN] No points to save")
        return
    
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision
    header.offsets = np.array([
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 2].min()),
    ])
    
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    if colors is not None and len(colors) == len(points):
        rgb16 = (np.clip(colors, 0.0, 1.0) * 65535).astype(np.uint16)
        las.red = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue = rgb16[:, 2]
    
    las.write(str(output_path))
    print(f"[OK] Saved {output_path} ({len(points):,} points)")


def main():
    parser = argparse.ArgumentParser(description="Merge multi-drone LiDAR data")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Data directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output LAS file (default: data/merged.las)")
    parser.add_argument("--drones", type=str, nargs="+", default=None,
                        help="Specific drone names to merge")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Voxel size for downsampling (0 = no downsampling)")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics only, don't save")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "merged.las"
    
    # Find drones
    drone_dirs = find_drone_dirs(data_dir, args.drones)
    
    if not drone_dirs:
        print("[ERROR] No drone data found")
        return 1
    
    print(f"[INFO] Found {len(drone_dirs)} drone(s): {[d.name for d in drone_dirs]}")
    
    # Load all points
    all_pts = []
    all_colors = []
    
    for drone_dir in drone_dirs:
        pts, colors = load_drone_points(drone_dir)
        
        if pts.size > 0:
            print(f"[INFO] {drone_dir.name}: {len(pts):,} points")
            all_pts.append(pts)
            if colors.size > 0:
                all_colors.append(colors)
    
    if not all_pts:
        print("[ERROR] No points loaded")
        return 1
    
    # Merge
    merged_pts = np.vstack(all_pts)
    merged_colors = np.vstack(all_colors) if all_colors else np.empty((0, 3))
    
    print(f"\n[INFO] Total before dedup: {len(merged_pts):,} points")
    
    # Bounds
    print(f"[INFO] X range: [{merged_pts[:,0].min():.2f}, {merged_pts[:,0].max():.2f}]")
    print(f"[INFO] Y range: [{merged_pts[:,1].min():.2f}, {merged_pts[:,1].max():.2f}]")
    print(f"[INFO] Z range: [{merged_pts[:,2].min():.2f}, {merged_pts[:,2].max():.2f}]")
    
    # Optional downsampling
    if args.voxel > 0:
        merged_pts, merged_colors = downsample_points(merged_pts, merged_colors, args.voxel)
        print(f"[INFO] After voxel downsampling ({args.voxel}m): {len(merged_pts):,} points")
    
    if args.stats:
        print("[INFO] Stats-only mode, not saving")
        return 0
    
    # Save
    save_las(output_path, merged_pts, merged_colors if merged_colors.size > 0 else None)
    
    return 0


if __name__ == "__main__":
    exit(main())