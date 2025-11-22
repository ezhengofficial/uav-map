import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class GridInfo:
    """Container for coverage grid metadata."""
    grid: np.ndarray        # bool (H, W) - True = seen
    wall_mask: np.ndarray   # bool (H, W) - True = wall
    x_min: float
    y_min: float
    width: int
    height: int
    cell_size: float
    
    def cell_to_enu(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid cell indices to ENU world coordinates (cell center)."""
        x = self.x_min + (ix + 0.5) * self.cell_size
        y = self.y_min + (iy + 0.5) * self.cell_size
        return x, y
    
    def enu_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert ENU coordinates to grid cell indices."""
        ix = int((x - self.x_min) / self.cell_size)
        iy = int((y - self.y_min) / self.cell_size)
        ix = np.clip(ix, 0, self.width - 1)
        iy = np.clip(iy, 0, self.height - 1)
        return ix, iy

    @property
    def coverage_ratio(self) -> float:
        """Fraction of cells that have been seen."""
        return float(self.grid.sum()) / max(self.grid.size, 1)


class CoverageGridBuilder:
    """Builds 2D coverage grids from 3D point clouds."""
    
    def __init__(self, cell_size: float = 0.5,
                 min_wall_points: int = 20,
                 min_wall_extent: float = 0.5,
                 min_passable_height: float = 4.0):
        self.cell_size = cell_size
        self.min_wall_points = min_wall_points
        self.min_wall_extent = min_wall_extent
        self.min_passable_height = min_passable_height
    
    def build(self, points_xyz: np.ndarray) -> GridInfo:
        """
        Build coverage grid and wall mask from XYZ points (ENU).
        
        Args:
            points_xyz: (N, 3) array of [x, y, z] in ENU meters
            
        Returns:
            GridInfo with coverage grid and wall mask
        """
        if points_xyz.size == 0:
            raise ValueError("No points provided")
        
        x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
        
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        
        eps = 1e-6
        width = int(np.ceil((x_max - x_min) / self.cell_size + eps))
        height = int(np.ceil((y_max - y_min) / self.cell_size + eps))
        
        # Ensure minimum grid size
        width = max(width, 1)
        height = max(height, 1)
        
        # Initialize arrays
        grid = np.zeros((height, width), dtype=bool)
        count = np.zeros((height, width), dtype=np.int32)
        z_min_arr = np.full((height, width), np.inf, dtype=np.float32)
        z_max_arr = np.full((height, width), -np.inf, dtype=np.float32)
        
        # Compute cell indices
        ix = ((x - x_min) / self.cell_size).astype(int)
        iy = ((y - y_min) / self.cell_size).astype(int)
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)
        
        # Populate grid
        grid[iy, ix] = True
        np.add.at(count, (iy, ix), 1)
        np.minimum.at(z_min_arr, (iy, ix), z)
        np.maximum.at(z_max_arr, (iy, ix), z)
        
        # Wall detection
        z_extent = np.where(count > 0, z_max_arr - z_min_arr, 0.0)
        wall_mask = (
            (count >= self.min_wall_points) & 
            (z_extent >= self.min_wall_extent) &
            (z_extent >= self.min_passable_height)
        )
        
        return GridInfo(
            grid=grid,
            wall_mask=wall_mask,
            x_min=x_min,
            y_min=y_min,
            width=width,
            height=height,
            cell_size=self.cell_size,
        )


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """4-connected binary dilation."""
    if iterations <= 0:
        return mask.copy()
    
    result = mask.copy()
    h, w = result.shape
    
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        grown = (
            padded[0:h, 1:w+1] |    # up
            padded[2:h+2, 1:w+1] |  # down
            padded[1:h+1, 0:w] |    # left
            padded[1:h+1, 2:w+2]    # right
        )
        result = result | grown
    
    return result


def find_frontiers(grid: np.ndarray) -> np.ndarray:
    """
    Find frontier cells: unknown cells adjacent to known cells.
    """
    h, w = grid.shape
    known = grid
    
    padded = np.pad(known, 1, constant_values=False)
    neighbor_known = (
        padded[0:h, 1:w+1] |    # up
        padded[2:h+2, 1:w+1] |  # down
        padded[1:h+1, 0:w] |    # left
        padded[1:h+1, 2:w+2]    # right
    )
    
    return (~known) & neighbor_known


def compute_deadzone_mask(grid_info: GridInfo, 
                          drone_pos_enu: np.ndarray,
                          radius: float) -> np.ndarray:
    """Create mask of cells within radius of drone position."""
    xs = grid_info.x_min + (np.arange(grid_info.width) + 0.5) * grid_info.cell_size
    ys = grid_info.y_min + (np.arange(grid_info.height) + 0.5) * grid_info.cell_size
    Xc, Yc = np.meshgrid(xs, ys)
    
    dx = Xc - float(drone_pos_enu[0])
    dy = Yc - float(drone_pos_enu[1])
    
    return (dx*dx + dy*dy) <= radius**2