import laspy
import numpy as np
import json
from pathlib import Path
from files_path import DATA_DIR, MERGED_LAS_FILE, SETTINGS_JSON_FILE, TRACKER_FILE

TRACKER_FILE = DATA_DIR / "final/merge_tracker.json"

# --- Tracker helpers ---
def _load_tracker() -> dict:
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_tracker(tracker: dict):
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f)

# --- Incremental merge ---
def _merge_las_files_incremental(input_dirs: list[Path], output_file: Path):
    """
    Incrementally merge LAS files from multiple drone directories.
    Assumes each LAS is already aligned in WORLD ENU coordinates.
    """
    tracker = _load_tracker()
    new_points, new_colors = [], []

    for d in input_dirs:
        drone_name = d.name
        last_idx = tracker.get(drone_name, -1)

        # Only consider files with index > last_idx
        new_files = []
        for f in sorted(d.glob("lidar_*.las")):
            try:
                idx = int(f.stem.split("_")[1])
            except Exception:
                continue
            if idx > last_idx:
                new_files.append((idx, f))

        if not new_files:
            continue

        for idx, f in new_files:
            if f.stat().st_size == 0:
                print(f"[WARN] Skipping empty LAS file: {f}")
                continue
            try:
                las = laspy.read(f)
            except Exception as e:
                print(f"[ERROR] Failed to read {f}: {e}")
                continue

            pts = np.vstack((las.x, las.y, las.z)).T
            new_points.append(pts)

            if hasattr(las, "red"):
                new_colors.append(np.vstack((las.red, las.green, las.blue)).T)

            tracker[drone_name] = idx  # update tracker

    if not new_points:
        print("[INFO] No new LAS files to merge")
        return

    new_points = np.vstack(new_points)
    new_colors = np.vstack(new_colors) if new_colors else None

    # Append to existing merged file if it exists
    if output_file.exists():
        las_out = laspy.read(output_file)
        old_pts = np.vstack((las_out.x, las_out.y, las_out.z)).T
        merged_points = np.vstack((old_pts, new_points))

        if new_colors is not None and hasattr(las_out, "red"):
            old_colors = np.vstack((las_out.red, las_out.green, las_out.blue)).T
            merged_colors = np.vstack((old_colors, new_colors))
        else:
            merged_colors = new_colors
    else:
        merged_points = new_points
        merged_colors = new_colors

    # Create merged LAS header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.min(merged_points, axis=0)

    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = merged_points[:,0], merged_points[:,1], merged_points[:,2]

    if merged_colors is not None:
        las_out.red, las_out.green, las_out.blue = merged_colors[:,0], merged_colors[:,1], merged_colors[:,2]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_file))
    _save_tracker(tracker)

    print(f"[OK] Incrementally merged into {output_file}")
    print(f"     Total points: {len(merged_points):,}")

def _merge_all_drones_to_final(data_root: Path, final_output: Path):
    drone_dirs = sorted([p for p in data_root.glob("Drone*") if p.is_dir()])
    if not drone_dirs:
        print(f"[WARNING] No Drone directories found in {data_root}")
        return
    print(f"[INFO] Incrementally merging LAS files from {len(drone_dirs)} Drone directories")
    _merge_las_files_incremental(drone_dirs, final_output)

def merge_data(data_root=None, output_file=None):
    if data_root is None:
        data_root = DATA_DIR
    if output_file is None:
        output_file = MERGED_LAS_FILE
    _merge_all_drones_to_final(data_root, output_file)
