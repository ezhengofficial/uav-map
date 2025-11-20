import argparse
from pathlib import Path
import laspy
import numpy as np
import sys
from ..config import DATA_DIR


# def find_logs_root(script_path: Path, explicit: Path | None) -> Path:
#     base = script_path.parents[1] / "data" / "logs"
#     if explicit:
#         return explicit
#     # if timestamped subfolders exist, pick newest
#     subs = [p for p in base.glob("*") if p.is_dir()]
#     if not subs:
#         return base
#     return max(subs, key=lambda p: p.stat().st_mtime)

def _merge_las_files(input_dirs: list[Path], output_file: Path):
    '''
    Merge all .las files from a list of directories into one LAS file.
    '''
    files = []
    for d in input_dirs:
        files.extend(sorted(d.glob("*.las")))

    # files = sorted(input_dir.glob("*.las"))
    if not files:
        print(f"[WARNING] No LAS files found in {input_dirs}")
        return
        # sys.exit(1)

    # print(f"[INFO] Found {len(files)} LAS files in {input_dir}")

    merged_points = []
    merged_colors = []

    first_header = None
    for f in files:
        las = laspy.read(f)
        # if first_header is None:
        #     first_header = las.header
        pts = np.vstack((las.x, las.y, las.z)).T
        merged_points.append(pts)
        if hasattr(las, "red"):
            merged_colors.append(np.vstack((las.red, las.green, las.blue)).T)
        # print(f"[INFO] Loaded {f.name}: {len(pts):,} points")

    merged_points = np.vstack(merged_points)
    merged_colors = np.vstack(merged_colors) if merged_colors else None

    # Create merged LAS header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([
        float(np.min(merged_points[:, 0])),
        float(np.min(merged_points[:, 1])),
        float(np.min(merged_points[:, 2])),
    ])

    las_out = laspy.LasData(header)
    las_out.x = merged_points[:, 0]
    las_out.y = merged_points[:, 1]
    las_out.z = merged_points[:, 2]

    if merged_colors is not None:
        las_out.red   = merged_colors[:, 0]
        las_out.green = merged_colors[:, 1]
        las_out.blue  = merged_colors[:, 2]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_file))
    print(f"[OK] Wrote merged LAS: {output_file}")
    print(f"     Total points: {len(merged_points):,}")

def _merge_all_drones_to_final(data_root: Path, final_output: Path):
    """
    Find all 'Drone <number>' directories under data_root,
    merge their LAS files into one global LAS file.
    """
    drone_dirs = sorted([p for p in data_root.glob("Drone *") if p.is_dir()])
    if not drone_dirs:
        print(f"[WARNING] No Drone directories found in {data_root}")
        return 

    print(f"[INFO] Merging LAS files from {len(drone_dirs)} Drone directories into {final_output}")
    _merge_las_files(drone_dirs, final_output)


def merge_data(data_root= None, output_file=None):
    if data_root is None:
        data_root = DATA_DIR
    if output_file is None:
        output_file = DATA_DIR / "final/merged.las"
    _merge_all_drones_to_final(data_root, output_file)


# def main():
#     parser = argparse.ArgumentParser(description="Merge multiple LAS files into a single LAS/LAZ file.")
#     parser.add_argument("--logs", type=str, default=None, help="Path to folder with LAS frames (default: latest in data/logs/).")
#     parser.add_argument("--outfile", type=str, default=None, help="Output .las or .laz file path.")
#     parser.add_argument("--compress", action="store_true", help="Write compressed LAZ output (requires laspy[laszip] or LAZ support).")
#     args = parser.parse_args()

#     script_path = Path(__file__).resolve()
#     input_dir = find_logs_root(script_path, Path(args.logs) if args.logs else None)

#     outname = args.outfile or (Path(__file__).resolve().parents[1] / "data" / "merged.las")
#     if args.compress and not str(outname).lower().endswith(".laz"):
#         outname = outname.with_suffix(".laz")

#     merge_las_files(input_dir, outname)

# if __name__ == "__main__":
#     main()
