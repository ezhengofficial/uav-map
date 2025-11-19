import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_frontiers(grid: np.ndarray) -> np.ndarray:
    """
    Frontier cells:
      - grid == False (unknown)
      - at least one 4-connected neighbor is True (known)
    Returns a boolean mask with same shape as grid.
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize coverage grid and frontier cells (plus waypoints if available)."
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Path to coverage_grid.npz (default: data/coverage_grid.npz).",
    )
    parser.add_argument(
        "--frontiers",
        type=str,
        default=None,
        help="Optional path to frontier_waypoints.npz (default: data/frontier_waypoints.npz if it exists).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save PNG instead of showing interactively.",
    )
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
    cell_size = float(data["cell_size"])

    h, w = grid.shape
    print(f"[INFO] Grid shape: {w} x {h}, cell_size={cell_size} m")

    # Compute frontier mask
    frontier_mask = find_frontiers(grid)
    num_frontiers = int(frontier_mask.sum())
    print(f"[INFO] Frontier cells: {num_frontiers}")

    # Check for frontier waypoint file
    if args.frontiers:
        wp_path = Path(args.frontiers)
    else:
        wp_path = script_path.parents[1] / "data" / "frontier_waypoints.npz"

    waypoints_enu = None
    if wp_path.exists():
        print(f"[INFO] Loading frontier waypoints: {wp_path}")
        wp_data = np.load(wp_path)
        if "waypoints_enu" in wp_data:
            waypoints_enu = wp_data["waypoints_enu"]  # (N,2) [x_enu, y_enu]
        else:
            print("[WARN] waypoints_enu not found in frontier_waypoints.npz; skipping waypoint overlay.")
    else:
        print(f"[INFO] No frontier waypoint file found at {wp_path}; skipping waypoint overlay.")

    # Build a visualization image
    # 0 = unknown, 1 = known, 2 = frontier
    vis = np.zeros_like(grid, dtype=np.uint8)
    vis[grid] = 1
    vis[frontier_mask] = 2

    # Colormap: dark gray for unknown, green for known, red for frontier
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([
        "#202020",  # 0 unknown
        "#55aa55",  # 1 known
        "#ff4444",  # 2 frontier
    ])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        vis,
        origin="lower",  # so row 0 is at the bottom, matching increasing Y
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_title("Coverage Grid (known / unknown / frontier)")
    ax.set_xlabel("X cell index")
    ax.set_ylabel("Y cell index")

    # Overlay waypoint dots if available
    if waypoints_enu is not None and waypoints_enu.size > 0:
        # Convert ENU positions to cell indices for overlay
        x_enu = waypoints_enu[:, 0]
        y_enu = waypoints_enu[:, 1]
        ix = (x_enu - x_min) / cell_size
        iy = (y_enu - y_min) / cell_size

        ax.scatter(
            ix,
            iy,
            s=20,
            marker="o",
            edgecolors="black",
            facecolors="none",
            linewidths=0.7,
            label="Frontier waypoints",
        )
        ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Unknown", "Known", "Frontier"])

    fig.tight_layout()

    if args.save:
        out_path = Path(args.save)
        fig.savefig(out_path, dpi=200)
        print(f"[OK] Saved coverage visualization to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
