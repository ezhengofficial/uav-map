# IMPORTANT: Set non-interactive backend BEFORE importing pyplot
# This fixes "main thread is not in main loop" errors with multi-threading
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, thread-safe

import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from exploration.coverage_grid import (
    CoverageGridBuilder, GridInfo, dilate_mask, compute_deadzone_mask
)
from exploration.frontier_planner import FrontierPlanner, PlanResult


class ExplorationConfig:
    """Configuration for exploration behavior."""
    def __init__(self):
        # Flight parameters
        self.altitude = 5.0           # meters (positive = up)
        self.speed = 5.0              # m/s
        self.scan_interval = 0.05     # seconds between LiDAR frames
        self.frames_per_scan = 15     # frames per exploration iteration
        
        # Coverage grid
        self.cell_size = 0.5          # meters per grid cell
        self.coverage_target = 0.85   # stop at 85% coverage
        self.max_iterations = 100
        
        # Navigation
        self.deadzone_radius = 15.0   # ignore frontiers within this radius
        self.escape_radius = 1.5      # carve bubble around drone
        self.max_step_cells = 50      # max grid cells per move
        
        # Safety bounds (NED) - these are global bounds
        self.x_min_ned = -150.0
        self.x_max_ned = 150.0
        self.y_min_ned = -150.0
        self.y_max_ned = 150.0
        
        # Collision handling
        self.step_dist = 1.0          # sub-step distance
        self.no_progress_thresh = 0.15
        self.collision_scans = 3


class Explorer:
    """
    Autonomous frontier-based exploration for a single drone.
    Supports region-constrained exploration for multi-drone partitioning.
    """
    
    def __init__(self, drone_agent, config: Optional[ExplorationConfig] = None):
        """
        Args:
            drone_agent: DroneAgent instance
            config: ExplorationConfig (uses defaults if None)
        """
        self.drone = drone_agent
        self.config = config or ExplorationConfig()
        
        # Build components
        self.grid_builder = CoverageGridBuilder(
            cell_size=self.config.cell_size
        )
        self.planner = FrontierPlanner(
            min_frontier_dist=self.config.deadzone_radius
        )
        
        # Debug output
        self.debug_dir = self.drone.data_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.iteration = 0
        self.grid_info: Optional[GridInfo] = None
        
        # Region constraint (set via set_region or from drone.assigned_region)
        self._region_ned: Optional[Tuple[float, float, float, float]] = None

    def set_region(self, x_min_ned: float, x_max_ned: float,
                   y_min_ned: float, y_max_ned: float):
        """
        Constrain this explorer to a specific region (NED coordinates).
        Drones will only explore frontiers within this region.
        """
        self._region_ned = (x_min_ned, x_max_ned, y_min_ned, y_max_ned)
        print(f"[{self.drone.name}] Assigned region NED: "
              f"X=[{x_min_ned:.1f}, {x_max_ned:.1f}], "
              f"Y=[{y_min_ned:.1f}, {y_max_ned:.1f}]")

    def _get_effective_bounds(self) -> Tuple[float, float, float, float]:
        """Get the effective bounds for this explorer (region or global)."""
        if self._region_ned is not None:
            return self._region_ned
        if self.drone.assigned_region is not None:
            return self.drone.assigned_region
        return (self.config.x_min_ned, self.config.x_max_ned,
                self.config.y_min_ned, self.config.y_max_ned)

    def run(self) -> float:
        """
        Execute exploration loop.
        Returns final coverage ratio.
        """
        print(f"[{self.drone.name}] Starting exploration...")
        
        # Ensure we have our own client connection (thread safety)
        self.drone.connect()
        
        # Initial setup
        self.drone.takeoff(self.config.altitude, speed=2.0)
        self.drone.estimate_floor()
        
        coverage = 0.0
        
        try:
            for it in range(self.config.max_iterations):
                self.iteration = it + 1
                print(f"\n[{self.drone.name}] ===== ITERATION {self.iteration}/{self.config.max_iterations} =====")
                
                # Phase 1: Scan
                self._scan_phase()
                
                # Phase 2: Build map
                pts_all = self.drone.load_all_points()
                if pts_all.shape[0] == 0:
                    print(f"[{self.drone.name}] No points yet, continuing...")
                    continue
                
                self.grid_info = self.grid_builder.build(pts_all)
                coverage = self._compute_region_coverage()
                print(f"[{self.drone.name}] Coverage (assigned region): {coverage*100:.2f}%")
                
                # Phase 3: Plan
                result = self._plan_phase()
                
                # Save debug visualization
                self._save_debug_plot()
                
                # Check termination
                if coverage >= self.config.coverage_target:
                    print(f"[{self.drone.name}] Coverage target reached: {coverage*100:.2f}%")
                    break
                
                if not result.success:
                    print(f"[{self.drone.name}] No reachable frontiers in assigned region")
                    break
                
                # Phase 4: Move
                collided = self._move_phase(result)
                if collided:
                    continue  # Rebuild map and replan
        
        finally:
            # Always land and disconnect
            self.drone.land()
            self.drone.disconnect()
        
        print(f"[{self.drone.name}] Exploration complete. Final coverage: {coverage*100:.2f}%")
        return coverage

    def _compute_region_coverage(self) -> float:
        """Compute coverage ratio within assigned region only."""
        if self.grid_info is None:
            return 0.0
        
        # Get region mask
        region_mask = self._build_region_mask()
        
        if region_mask is None:
            return self.grid_info.coverage_ratio
        
        # Count covered cells within region
        covered_in_region = int((self.grid_info.grid & region_mask).sum())
        total_in_region = int(region_mask.sum())
        
        if total_in_region == 0:
            return 0.0
        
        return covered_in_region / total_in_region

    def _build_region_mask(self) -> Optional[np.ndarray]:
        """Build mask of cells within assigned region."""
        if self.grid_info is None:
            return None
        
        bounds = self._get_effective_bounds()
        x_min_ned, x_max_ned, y_min_ned, y_max_ned = bounds
        
        # Grid cell centers in ENU
        h, w = self.grid_info.grid.shape
        xs_enu = self.grid_info.x_min + (np.arange(w) + 0.5) * self.grid_info.cell_size
        ys_enu = self.grid_info.y_min + (np.arange(h) + 0.5) * self.grid_info.cell_size
        Xs_enu, Ys_enu = np.meshgrid(xs_enu, ys_enu)
        
        # Convert to NED: X_ned = Y_enu, Y_ned = X_enu
        Xs_ned = Ys_enu
        Ys_ned = Xs_enu
        
        # Check bounds
        in_region = (
            (Xs_ned >= x_min_ned) & (Xs_ned <= x_max_ned) &
            (Ys_ned >= y_min_ned) & (Ys_ned <= y_max_ned)
        )
        
        return in_region

    def _scan_phase(self):
        """Capture multiple LiDAR frames."""
        for _ in range(self.config.frames_per_scan):
            self.drone.scan_and_save()
            time.sleep(self.config.scan_interval)

    def _plan_phase(self) -> PlanResult:
        """Build obstacle map and plan path to frontier within region."""
        if self.grid_info is None:
            return PlanResult(None, None, False)
        
        # Get drone position
        pos_enu = self.drone.get_position_enu()[:2]
        
        # Build obstacle mask
        obstacle_mask = self.planner.build_obstacle_mask(self.grid_info)
        
        # Carve escape bubble
        obstacle_mask = self._carve_escape_bubble(obstacle_mask, pos_enu)
        
        # Find valid frontiers
        deadzone = compute_deadzone_mask(
            self.grid_info, pos_enu, self.config.deadzone_radius
        )
        frontier_mask = self.planner.find_valid_frontiers(
            self.grid_info, obstacle_mask, deadzone
        )
        
        # IMPORTANT: Mask frontiers outside assigned region
        region_mask = self._build_region_mask()
        if region_mask is not None:
            frontier_mask = frontier_mask & region_mask
        
        num_frontiers = int(frontier_mask.sum())
        print(f"[{self.drone.name}] Valid frontiers in region: {num_frontiers}")
        
        if num_frontiers == 0:
            return PlanResult(None, None, False)
        
        # Plan path
        return self.planner.plan(
            self.grid_info, obstacle_mask, frontier_mask, pos_enu
        )

    def _carve_escape_bubble(self, obstacle_mask: np.ndarray, 
                             pos_enu: np.ndarray) -> np.ndarray:
        """Clear obstacles around drone to prevent trapping."""
        if self.grid_info is None:
            return obstacle_mask
        
        bubble = compute_deadzone_mask(
            self.grid_info, pos_enu, self.config.escape_radius
        )
        result = obstacle_mask.copy()
        result[bubble] = False
        return result

    def _move_phase(self, plan: PlanResult) -> bool:
        """
        Move along planned path.
        Returns True if collision detected.
        """
        if plan.path is None or len(plan.path) <= 1:
            return False
        
        # Pick waypoint along path
        step_idx = min(1 + self.config.max_step_cells, len(plan.path) - 1)
        iy, ix = plan.path[step_idx]
        
        # Convert to ENU
        x_enu, y_enu = self.grid_info.cell_to_enu(ix, iy)
        
        # Convert to NED
        x_ned = y_enu  # N = Y_enu
        y_ned = x_enu  # E = X_enu
        z_ned = -self.config.altitude
        
        # Clamp to assigned region bounds
        bounds = self._get_effective_bounds()
        x_ned = np.clip(x_ned, bounds[0], bounds[1])
        y_ned = np.clip(y_ned, bounds[2], bounds[3])
        
        print(f"[{self.drone.name}] Target: ENU({x_enu:.1f}, {y_enu:.1f}) -> NED({x_ned:.1f}, {y_ned:.1f})")
        
        # Execute move with collision detection
        return self._safe_move(x_ned, y_ned, z_ned)

    def _safe_move(self, x_ned: float, y_ned: float, z_ned: float) -> bool:
        """
        Move with progress-based collision detection.
        Returns True if stuck/collision detected.
        """
        pos_before = self.drone.get_position_ned()[:2]
        target = np.array([x_ned, y_ned])
        
        dist = float(np.linalg.norm(target - pos_before))
        if dist < 0.3:
            return False
        
        # Move in steps
        direction = (target - pos_before) / dist
        step_len = min(self.config.step_dist, dist)
        n_steps = int(np.ceil(dist / step_len))
        
        for k in range(1, n_steps + 1):
            step_pos = pos_before + direction * min(k * step_len, dist)
            
            timeout = max(1.0, 2.0 * step_len / self.config.speed)
            self.drone.client.moveToPositionAsync(
                float(step_pos[0]), float(step_pos[1]), z_ned,
                self.config.speed, timeout_sec=timeout,
                vehicle_name=self.drone.name
            ).join()
            
            # Check progress
            pos_after = self.drone.get_position_ned()[:2]
            actual_move = float(np.linalg.norm(pos_after - pos_before))
            
            if actual_move < self.config.no_progress_thresh:
                print(f"[{self.drone.name}] No progress detected, rescanning")
                for _ in range(self.config.collision_scans):
                    self.drone.scan_and_save()
                    time.sleep(self.config.scan_interval)
                return True
            
            pos_before = pos_after
        
        return False

    def _save_debug_plot(self):
        """Save visualization of current map state."""
        if self.grid_info is None:
            return
        
        obstacle_mask = self.planner.build_obstacle_mask(self.grid_info)
        frontier_mask = self.planner.find_valid_frontiers(
            self.grid_info, obstacle_mask
        )
        
        # Mask frontiers to region
        region_mask = self._build_region_mask()
        if region_mask is not None:
            frontier_mask = frontier_mask & region_mask
        
        # Build visualization array
        vis = np.zeros_like(self.grid_info.grid, dtype=np.uint8)
        vis[self.grid_info.grid] = 1
        vis[frontier_mask] = 2
        vis[self.grid_info.wall_mask] = 3
        vis[obstacle_mask] = 4
        
        # Mark region boundary
        if region_mask is not None:
            boundary = region_mask ^ dilate_mask(region_mask, 1)
            vis[boundary] = 5
        
        cmap = ListedColormap([
            "#202020",  # unknown
            "#55aa55",  # known
            "#ff4444",  # frontier
            "#3366ff",  # wall
            "#ffcc00",  # obstacle
            "#ff00ff",  # region boundary
        ])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(vis, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(f"{self.drone.name} - Coverage (iter {self.iteration})")
        
        # Plot drone position
        pos = self.drone.get_position_enu()[:2]
        ix, iy = self.grid_info.enu_to_cell(pos[0], pos[1])
        ax.scatter(ix, iy, s=40, marker="x", color="cyan")
        
        fig.tight_layout()
        out_path = self.debug_dir / f"coverage_{self.iteration:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)