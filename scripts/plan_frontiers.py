import argparse
from pathlib import Path
import numpy as np


def find_frontiers(grid: np.ndarray) -> np.ndarray:
    """
    Find frontier cells:
      - grid == False (unknown)
      - at least one 4-connected neighbor is True (known)
    Returns a boolean mask of the same shape.
    """
    h, w = grid.shape
    known = grid

    pad = np.pad(known, pad_width=1, mode="constant", constant_values=False)
    up    = pad[0:h,   1:w+1]
    down  = pad[2:h+2, 1:w+1]
    left  = pad[1:h+1, 0:w]
    right = pad[1:h+1, 2:w+2]

    neighbor_known = up | down | left | right
    frontiers = (~known) & neighbor_known
    return frontiers


def dilate_frontier(frontier_mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Expand frontier cells by N 4-connected dilation steps (pure NumPy).
    """
    f = frontier_mask.copy()
    h, w = f.shape

    for _ in range(iterations):
        padded = np.pad(f, 1, constant_values=False)

        up    = padded[0:h,   1:w+1]
        down  = padded[2:h+2, 1:w+1]
        left  = padded[1:h+1, 0:w]
        right = padded[1:h+1, 2:w+2]

        grown = up | down | left | right
        f = f | grown

    return f


def sample_frontiers(frontier_mask: np.ndarray, max_points: int = 100) -> np.ndarray:
    """
    Given a boolean frontier mask, return up to max_points (iy, ix) indices.
    Simple strategy: take all, then subsample uniformly if too many.
    """
    ys, xs = np.where(frontier_mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)

    idxs = np.arange(len(xs))
    if len(idxs) > max_points:
        rng = np.random.default_rng()
        idxs = rng.choice(idxs, size=max_points, replace=False)

    return np.stack([ys[idxs], xs[idxs]], axis=1)  # (N, 2) [iy, ix]


def main():
    parser = argparse.ArgumentParser(description="Plan frontier waypoints from coverage grid.")
    parser.add_argument("--grid", type=str, default=None,
                        help="Path to coverage_grid.npz (default: data/coverage_grid.npz).")
    parser.add_argument("--max-frontiers", type=int, default=50,
                        help="Maximum number of frontier waypoints to output (default: 50).")
    parser.add_argument("--cell", type=float, default=None,
                        help="Override cell size if desired (default: use saved value).")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_grid = script_path.parents[1] / "data" / "coverage_grid.npz"
    grid_path = Path(args.grid) if args.grid else default_grid

    if not grid_path.exists():
        raise SystemExit(f"[ERROR] Coverage grid not found: {grid_path}")

    print(f"[INFO] Loading coverage grid: {grid_path}")
    data = np.load(grid_path)
    grid = data["grid"].astype(bool)
    x_min = float(data["x_min"])
    y_min = float(data["y_min"])
    cell_size = float(args.cell) if args.cell is not None else float(data["cell_size"])

    h, w = grid.shape
    print(f"[INFO] Grid shape: {w} x {h}, cell_size={cell_size} m")

    frontier_mask = find_frontiers(grid)
    print(f"[INFO] Raw frontier cells: {int(frontier_mask.sum())}")

    # Dilate frontiers to make them thicker / easier targets
    frontier_mask = dilate_frontier(frontier_mask, iterations=2)
    num_frontier_cells = int(frontier_mask.sum())
    print(f"[INFO] Dilated frontier cells: {num_frontier_cells}")

    if num_frontier_cells == 0:
        print("[WARN] No frontiers found. Coverage may already be complete.")
        out_path = script_path.parents[1] / "data" / "frontier_waypoints.npz"
        np.savez_compressed(out_path,
                            waypoints_enu=np.zeros((0, 2), dtype=np.float32),
                            waypoints_ned=np.zeros((0, 2), dtype=np.float32))
        print(f"[OK] Saved empty frontier list to: {out_path}")
        return

    sampled = sample_frontiers(frontier_mask, max_points=args.max_frontiers)
    print(f"[INFO] Sampled {len(sampled)} frontier cells for waypoints.")

    iy = sampled[:, 0]
    ix = sampled[:, 1]

    x_enu = x_min + (ix + 0.5) * cell_size
    y_enu = y_min + (iy + 0.5) * cell_size

    waypoints_enu = np.stack([x_enu, y_enu], axis=1).astype(np.float32)

    # Convert ENU -> AirSim NED (X_ned=N=y_enu, Y_ned=E=x_enu)
    x_ned = y_enu
    y_ned = x_enu
    waypoints_ned = np.stack([x_ned, y_ned], axis=1).astype(np.float32)

    out_path = script_path.parents[1] / "data" / "frontier_waypoints.npz"
    np.savez_compressed(out_path,
                        waypoints_enu=waypoints_enu,
                        waypoints_ned=waypoints_ned,
                        cell_size=cell_size,
                        x_min=x_min,
                        y_min=y_min)
    print(f"[OK] Saved frontier waypoints to: {out_path}")
    print("[INFO] First few waypoints (ENU):")
    for i in range(min(5, len(waypoints_enu))):
        print(f"  ENU ({waypoints_enu[i,0]:.2f}, {waypoints_enu[i,1]:.2f})"
              f"  | NED ({waypoints_ned[i,0]:.2f}, {waypoints_ned[i,1]:.2f})")


if __name__ == "__main__":
    main()
