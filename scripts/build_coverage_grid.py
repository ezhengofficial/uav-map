import argparse
from pathlib import Path
import numpy as np
import laspy

# ============================================
# Adjustable global constant for grid cell size
# ============================================
CELL_SIZE_DEFAULT = 2.0     # <--- CHANGE THIS to adjust resolution
# ============================================


def build_coverage_grid(points_xy: np.ndarray, cell_size: float):
    """
    Build a 2D coverage grid from XY points.

    points_xy: (N, 2) array of [x_enu, y_enu] in meters.
    cell_size: edge length of each square cell in meters.
    """
    if points_xy.size == 0:
        raise ValueError("No points provided for coverage grid.")

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    eps = 1e-6
    width  = int(np.ceil((x_max - x_min) / cell_size + eps))
    height = int(np.ceil((y_max - y_min) / cell_size + eps))

    grid = np.zeros((height, width), dtype=bool)

    ix = ((x - x_min) / cell_size).astype(int)
    iy = ((y - y_min) / cell_size).astype(int)

    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)

    grid[iy, ix] = True

    return grid, x_min, y_min, width, height


def main():
    parser = argparse.ArgumentParser(description="Build a 2D coverage grid from merged LAS.")
    parser.add_argument("--las", type=str, default=None,
                        help="Path to merged LAS/LAZ file (default: data/merged.las).")
    parser.add_argument("--cell", type=float, default=None,
                        help=f"Cell size in meters (default: {CELL_SIZE_DEFAULT}).")
    parser.add_argument("--z-min", type=float, default=None,
                        help="Optional min Z (ENU) filter.")
    parser.add_argument("--z-max", type=float, default=None,
                        help="Optional max Z (ENU) filter.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_las = script_path.parents[1] / "data" / "merged.las"
    las_path = Path(args.las) if args.las else default_las

    if not las_path.exists():
        raise SystemExit(f"[ERROR] LAS file not found: {las_path}")

    print(f"[INFO] Loading LAS: {las_path}")
    las = laspy.read(las_path)

    try:
        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to read x/y/z from LAS: {e!r}")
        print(f"        point_format: {las.header.point_format.id}")
        print(f"        has x: {hasattr(las, 'x')}, has X: {hasattr(las, 'X')}")
        raise

    pts = np.column_stack((x, y, z))

    # Optional Z filtering
    if args.z_min is not None:
        pts = pts[pts[:, 2] >= args.z_min]
    if args.z_max is not None:
        pts = pts[pts[:, 2] <= args.z_max]

    if pts.shape[0] == 0:
        raise SystemExit("[ERROR] No points left after Z filtering.")

    # Use override or global default
    cell_size = args.cell if args.cell is not None else CELL_SIZE_DEFAULT

    grid, x_min, y_min, width, height = build_coverage_grid(pts[:, :2], cell_size)

    seen_cells   = int(grid.sum())
    total_cells  = grid.size
    coverage_pct = 100.0 * seen_cells / max(total_cells, 1)

    print(f"[INFO] Grid size: {width} x {height} cells (cell_size = {cell_size} m)")
    print(f"[INFO] World X range: [{x_min:.2f}, {x_min + width*cell_size:.2f}] m")
    print(f"[INFO] World Y range: [{y_min:.2f}, {y_min + height*cell_size:.2f}] m")
    print(f"[INFO] Seen cells: {seen_cells} / {total_cells} ({coverage_pct:.2f}%)")

    out_path = script_path.parents[1] / "data" / "coverage_grid.npz"
    np.savez_compressed(
        out_path,
        grid=grid,
        x_min=x_min,
        y_min=y_min,
        cell_size=cell_size,
    )
    print(f"[OK] Saved coverage grid to: {out_path}")


if __name__ == "__main__":
    main()
