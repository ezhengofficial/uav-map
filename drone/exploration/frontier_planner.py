"""
Frontier detection and A* path planning.
Extracted from explore_single_loop.py for modularity.
"""
import numpy as np
from heapq import heappush, heappop
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

from exploration.coverage_grid import GridInfo, find_frontiers, dilate_mask


@dataclass 
class PlanResult:
    """Result of path planning."""
    path: Optional[List[Tuple[int, int]]]  # List of (iy, ix) cells
    goal_cell: Optional[Tuple[int, int]]   # Target frontier cell
    success: bool


class FrontierPlanner:
    """Plans paths to unexplored frontier regions."""
    
    def __init__(self,
                 min_frontier_dist: float = 3.0,  # Reduced - was too large
                 min_frontier_neighbors: int = 3,  # Reduced for sensitivity
                 frontier_dilate_iters: int = 1,   # Slight dilation
                 wall_dilate_iters: int = 1,       # Reduced - was blocking paths
                 obstacle_dilate_iters: int = 1,
                 max_candidates: int = 200):       # Check more candidates
        self.min_frontier_dist = min_frontier_dist
        self.min_frontier_neighbors = min_frontier_neighbors
        self.frontier_dilate_iters = frontier_dilate_iters
        self.wall_dilate_iters = wall_dilate_iters
        self.obstacle_dilate_iters = obstacle_dilate_iters
        self.max_candidates = max_candidates
        
        # Track blocked goals to avoid getting stuck
        self.blocked_goals: Set[Tuple[int, int]] = set()
        self.last_goal: Optional[Tuple[int, int]] = None
        self.goal_repeat_count: int = 0
        self.max_repeats: int = 10

    def build_obstacle_mask(self, grid_info: GridInfo) -> np.ndarray:
        """Build navigation obstacle mask from walls."""
        wall_mask = dilate_mask(grid_info.wall_mask, self.wall_dilate_iters)
        return dilate_mask(wall_mask, self.obstacle_dilate_iters)

    def find_valid_frontiers(self, 
                             grid_info: GridInfo,
                             obstacle_mask: np.ndarray,
                             deadzone_mask: Optional[np.ndarray] = None
                             ) -> np.ndarray:
        """
        Find frontier cells that are valid navigation targets.
        """
        frontier_mask = find_frontiers(grid_info.grid)
        
        # Dilate frontiers if configured
        if self.frontier_dilate_iters > 0:
            frontier_mask = dilate_mask(frontier_mask, self.frontier_dilate_iters)
        
        # Remove frontiers in obstacles
        frontier_mask &= ~obstacle_mask
        
        # Remove frontiers in deadzone
        if deadzone_mask is not None:
            frontier_mask &= ~deadzone_mask
        
        # Filter tiny frontiers (pinholes)
        frontier_mask = self._filter_tiny_frontiers(frontier_mask, grid_info.grid)
        
        return frontier_mask

    def _filter_tiny_frontiers(self, frontier_mask: np.ndarray, 
                                grid: np.ndarray) -> np.ndarray:
        """Remove frontier cells with too few unknown neighbors."""
        h, w = grid.shape
        unknown = (~grid).astype(np.int32)
        
        # Count unknown cells in 3x3 neighborhood
        p = np.pad(unknown, 1, constant_values=0)
        local_unknown = (
            p[0:h, 0:w] + p[0:h, 1:w+1] + p[0:h, 2:w+2] +
            p[1:h+1, 0:w] + p[1:h+1, 1:w+1] + p[1:h+1, 2:w+2] +
            p[2:h+2, 0:w] + p[2:h+2, 1:w+1] + p[2:h+2, 2:w+2]
        )
        
        return frontier_mask & (local_unknown >= self.min_frontier_neighbors)

    def plan(self, 
             grid_info: GridInfo,
             obstacle_mask: np.ndarray,
             frontier_mask: np.ndarray,
             drone_pos_enu: np.ndarray) -> PlanResult:
        """
        Plan a path to the nearest reachable frontier.
        """
        # Get drone's grid cell
        sx, sy = grid_info.enu_to_cell(drone_pos_enu[0], drone_pos_enu[1])
        start = (sy, sx)
        
        # Get candidate frontier cells
        ys, xs = np.where(frontier_mask & ~obstacle_mask)
        if len(xs) == 0:
            return PlanResult(None, None, False)
        
        # Compute distances to frontiers
        x_enu = grid_info.x_min + (xs + 0.5) * grid_info.cell_size
        y_enu = grid_info.y_min + (ys + 0.5) * grid_info.cell_size
        
        dx = x_enu - drone_pos_enu[0]
        dy = y_enu - drone_pos_enu[1]
        distances = np.sqrt(dx*dx + dy*dy)
        
        # Filter by minimum distance
        valid = distances >= self.min_frontier_dist
        if not np.any(valid):
            return PlanResult(None, None, False)
        
        # Sort by distance (nearest first)
        valid_idxs = np.where(valid)[0]
        sorted_idxs = valid_idxs[np.argsort(distances[valid_idxs])]
        
        # Try A* to each candidate
        for idx in sorted_idxs[:self.max_candidates]:
            goal = (ys[idx], xs[idx])
            
            if goal in self.blocked_goals:
                continue
            
            path = self._astar(obstacle_mask, start, goal)
            if path is not None:
                # Track repeated goals
                if goal == self.last_goal:
                    self.goal_repeat_count += 1
                    if self.goal_repeat_count > self.max_repeats:
                        self.blocked_goals.add(goal)
                        continue
                else:
                    self.last_goal = goal
                    self.goal_repeat_count = 1
                
                return PlanResult(path, goal, True)
        
        return PlanResult(None, None, False)

    def _astar(self, obstacle_mask: np.ndarray,
               start: Tuple[int, int],
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding on 4-connected grid.
        
        Args:
            obstacle_mask: True = blocked
            start: (iy, ix)
            goal: (iy, ix)
        """
        h, w = obstacle_mask.shape
        sy, sx = start
        gy, gx = goal
        
        # Validate bounds
        if not (0 <= sx < w and 0 <= sy < h):
            return None
        if not (0 <= gx < w and 0 <= gy < h):
            return None
        if obstacle_mask[gy, gx]:
            return None
        
        def heuristic(y, x):
            return abs(y - gy) + abs(x - gx)
        
        open_set = [(0, (sy, sx))]
        came_from = {(sy, sx): None}
        g_score = {(sy, sx): 0}
        
        while open_set:
            _, (cy, cx) = heappop(open_set)
            
            if (cy, cx) == (gy, gx):
                # Reconstruct path
                path = []
                cur = (cy, cx)
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
                return path
            
            for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if obstacle_mask[ny, nx]:
                    continue
                
                tentative_g = g_score[(cy, cx)] + 1
                
                if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                    g_score[(ny, nx)] = tentative_g
                    f = tentative_g + heuristic(ny, nx)
                    heappush(open_set, (f, (ny, nx)))
                    came_from[(ny, nx)] = (cy, cx)
        
        return None

    def reset_blocks(self):
        """Clear blocked goals (call when exploration restarts)."""
        self.blocked_goals.clear()
        self.last_goal = None
        self.goal_repeat_count = 0