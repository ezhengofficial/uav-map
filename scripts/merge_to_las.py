import argparse, glob
from pathlib import Path
import numpy as np

try:
    import laspy
except ImportError as e:
    raise SystemExit(
        "laspy is required. Install with:\n  pip install laspy\n"
    ) from e

# ---------- Defaults ----------
DEFAULT_VOXEL = 0.10   # meters
DEFAULT_DENOISE = True
DEFAULT_COLORIZE = True
DEFAULT_OUT = "data/merged.las"
# -----------------------------

def find_logs_root(script_path: Path, explicit: Path | None) -> Path:
    base = script_path.parents[1] / "data" / "logs"
    if explicit:
        return explicit
    # if timestamped subfolders exist, pick newest
    subs = [p for p in base.glob("*") if p.is_dir()]
    if not subs:
        return base
    return max(subs, key=lambda p: p.stat().st_mtime)

def load_frames(logdir: Path) -> list[np.ndarray]:
    files = sorted(glob.glob(str(logdir / "lidar_*.npy")))
    if not files:
        raise SystemExit(f"[ERROR] No lidar_*.npy found in {logdir}")
    clouds = []
    for f in files:
        pts = np.load(f)
        pts = pts.reshape(-1, 3).astype(np.float32)
        clouds.append(pts)
    return clouds

def voxel_downsample_numpy(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0 or len(points) == 0:
        return points
    grid = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return points[idx]

def radius_outlier_filter_numpy(points: np.ndarray, radius=0.30, nb_points=8) -> np.ndarray:
    if len(points) == 0:
        return points
    voxel = radius / 2.0
    grid = np.floor(points / voxel).astype(np.int64)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, g in enumerate(map(tuple, grid)):
        buckets[g].append(i)
    keep = np.zeros(len(points), dtype=bool)
    off = [-1, 0, 1]
    nbr_offsets = [(dx,dy,dz) for dx in off for dy in off for dz in off]
    r2 = radius * radius
    for g, idxs in buckets.items():
        cand = []
        gx, gy, gz = g
        for d in nbr_offsets:
            cand.extend(buckets.get((gx+d[0], gy+d[1], gz+d[2]), []))
        if not cand: 
            continue
        cand_pts = points[cand]
        pts_block = points[idxs]
        for j, p in zip(idxs, pts_block):
            d2 = np.sum((cand_pts - p)**2, axis=1)
            if np.count_nonzero(d2 <= r2) >= nb_points:
                keep[j] = True
    return points[keep]

def colorize_by_height(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = points[:, 2]
    zmin, zmax = float(np.min(z)), float(np.max(z))
    norm = (z - zmin) / max(zmax - zmin, 1e-6)
    # simple gradient R (high) / G mid / B (low)
    return np.stack([norm, 0.5 * np.ones_like(norm), 1.0 - norm], axis=1).astype(np.float32)

def write_las(path: Path, points: np.ndarray, colors01: np.ndarray | None):
    path.parent.mkdir(parents=True, exist_ok=True)

    # LAS 1.2, Point Format 3 (XYZ + RGB)
    hdr = laspy.LasHeader(point_format=3, version="1.2")

    # Choose scales/offsets (LAS needs integer quantization)
    # 1 cm quantization is fine for sim:
    hdr.scales = np.array([0.01, 0.01, 0.01])
    hdr.offsets = np.array([
        float(np.min(points[:,0], initial=0.0)),
        float(np.min(points[:,1], initial=0.0)),
        float(np.min(points[:,2], initial=0.0)),
    ])

    las = laspy.LasData(hdr)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if colors01 is not None and len(colors01) == len(points):
        # LAS stores RGB as uint16 (0..65535)
        rgb16 = (np.clip(colors01, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
        las.red   = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue  = rgb16[:, 2]

    las.write(str(path))
    print(f"[OK] Wrote {path}  ({len(points):,} points)")

def main():
    parser = argparse.ArgumentParser(description="Merge LiDAR frames and write LAS (.las) for viewing in LAS/LAZ viewers.")
    parser.add_argument("--logs", type=str, default=None, help="Path to logs dir (default: latest subfolder in data/logs or data/logs).")
    parser.add_argument("--voxel", type=float, default=DEFAULT_VOXEL, help="Voxel downsample size in meters (default: 0.10).")
    parser.add_argument("--no-denoise", action="store_true", help="Disable simple radius outlier removal.")
    parser.add_argument("--no-color", action="store_true", help="Disable colorization by height (writes XYZ only).")
    parser.add_argument("--outfile", type=str, default=str(Path(__file__).resolve().parents[1] / DEFAULT_OUT), help="Output LAS path (default: data/merged.las)")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    logdir = find_logs_root(script_path, Path(args.logs) if args.logs else None)
    print(f"[INFO] Using logs from: {logdir}")

    clouds = load_frames(logdir)
    print(f"[INFO] Loaded {len(clouds)} frames")
    pts = np.concatenate(clouds, axis=0)
    print(f"[INFO] Total points before filters: {pts.shape[0]:,}")

    if args.voxel > 0:
        pts = voxel_downsample_numpy(pts, args.voxel)
        print(f"[INFO] After voxel({args.voxel:.2f} m): {pts.shape[0]:,}")

    if not args.no_denoise:
        pts = radius_outlier_filter_numpy(pts, radius=0.30, nb_points=8)
        print(f"[INFO] After denoise: {pts.shape[0]:,}")

    colors = None if args.no_color else colorize_by_height(pts)

    out = Path(args.outfile)
    write_las(out, pts, colors)

if __name__ == "__main__":
    main()
