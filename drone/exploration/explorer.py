"""
Exploration with FIXED region calculation and logging to file.
"""
import matplotlib
matplotlib.use('Agg')

import time
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, TextIO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import threading

from exploration.coverage_grid import (
    CoverageGridBuilder, GridInfo, dilate_mask, compute_deadzone_mask
)
from exploration.frontier_planner import FrontierPlanner, PlanResult


class SharedCoverageData:
    """Thread-safe shared coverage data."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data_lock = threading.Lock()
            cls._instance._all_points = {}
        return cls._instance
    
    def update_points(self, name: str, points: np.ndarray):
        with self._data_lock:
            self._all_points[name] = points.copy()
    
    def get_all_points(self) -> np.ndarray:
        with self._data_lock:
            if not self._all_points:
                return np.empty((0, 3), dtype=np.float32)
            return np.vstack(list(self._all_points.values()))
    
    def get_global_coverage(self, cell_size: float = 1.0) -> float:
        pts = self.get_all_points()
        if pts.size == 0:
            return 0.0
        x, y = pts[:, 0], pts[:, 1]
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        width = max(1, int(np.ceil((x_max - x_min) / cell_size)))
        height = max(1, int(np.ceil((y_max - y_min) / cell_size)))
        grid = np.zeros((height, width), dtype=bool)
        ix = np.clip(((x - x_min) / cell_size).astype(int), 0, width - 1)
        iy = np.clip(((y - y_min) / cell_size).astype(int), 0, height - 1)
        grid[iy, ix] = True
        return float(grid.sum()) / grid.size


class ExplorationConfig:
    def __init__(self):
        self.altitude = 5.0
        self.speed = 3.0
        self.scan_interval = 0.1
        self.frames_per_scan = 10
        self.cell_size = 1.0
        self.coverage_target = 0.85
        self.max_iterations = 100
        self.deadzone_radius = 3.0
        self.escape_radius = 2.0
        self.max_step_cells = 15
        self.min_move_dist = 2.0
        self.x_min_ned = -150.0
        self.x_max_ned = 150.0
        self.y_min_ned = -150.0
        self.y_max_ned = 150.0
        self.step_dist = 2.0
        self.no_progress_thresh = 0.3
        self.collision_scans = 3
        self.move_timeout = 15.0
        self.stale_iterations_limit = 5
        self.min_frontier_count = 3
        self.save_debug_plots = True
        self.verbose = True
        self.use_shared_coverage = True
        
        # Logging
        self.log_to_file = True


class DroneLogger:
    """Redirect drone output to file."""
    def __init__(self, drone_name: str, debug_dir: Path):
        self.drone_name = drone_name
        self.log_file: Optional[TextIO] = None
        
        if debug_dir:
            log_path = debug_dir / f"{drone_name}_log.txt"
            self.log_file = open(log_path, 'w', buffering=1)
    
    def log(self, msg: str):
        line = f"[{self.drone_name}] {msg}"
        if self.log_file:
            self.log_file.write(line + "\n")
        # Also print to console but with drone prefix
        print(line)
    
    def close(self):
        if self.log_file:
            self.log_file.close()


class Explorer:
    """Autonomous exploration with proper region handling."""
    
    def __init__(self, drone_agent, config: Optional[ExplorationConfig] = None):
        self.drone = drone_agent
        self.config = config or ExplorationConfig()
        
        self.grid_builder = CoverageGridBuilder(cell_size=self.config.cell_size)
        self.planner = FrontierPlanner(min_frontier_dist=self.config.deadzone_radius)
        
        self.debug_dir = self.drone.data_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = DroneLogger(self.drone.name, self.debug_dir) if self.config.log_to_file else None
        
        self.iteration = 0
        self.grid_info: Optional[GridInfo] = None
        
        # Region in WORLD ENU coordinates (not NED!)
        self._region_enu: Optional[Tuple[float, float, float, float]] = None
        
        # Progress tracking
        self._last_position: Optional[np.ndarray] = None
        self._total_distance = 0.0
        self._last_coverage = 0.0
        self._stale_count = 0
        self._visited_cells: set = set()
        
        self._shared = SharedCoverageData() if self.config.use_shared_coverage else None

    def _log(self, msg: str):
        if self.logger:
            self.logger.log(msg)
        elif self.config.verbose:
            print(f"[{self.drone.name}] {msg}")

    def set_region_enu(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set region in WORLD ENU coordinates."""
        self._region_enu = (x_min, x_max, y_min, y_max)
        self._log(f"Region ENU: X=[{x_min:.0f},{x_max:.0f}] Y=[{y_min:.0f},{y_max:.0f}]")

    def set_region_ned(self, x_min_ned: float, x_max_ned: float, 
                       y_min_ned: float, y_max_ned: float):
        """Set region from NED bounds (converts to ENU internally)."""
        # NED to ENU: X_enu = Y_ned, Y_enu = X_ned
        x_min_enu = min(y_min_ned, y_max_ned)
        x_max_enu = max(y_min_ned, y_max_ned)
        y_min_enu = min(x_min_ned, x_max_ned)
        y_max_enu = max(x_min_ned, x_max_ned)
        
        self._region_enu = (x_min_enu, x_max_enu, y_min_enu, y_max_enu)
        self._log(f"Region NED->ENU: X=[{x_min_enu:.0f},{x_max_enu:.0f}] Y=[{y_min_enu:.0f},{y_max_enu:.0f}]")

    def run(self) -> float:
        self._log("========== STARTING ==========")
        
        self.drone.connect()
        self.drone.takeoff(self.config.altitude, speed=2.0)
        time.sleep(1.0)
        self.drone.estimate_floor()
        
        self._last_position = self.drone.get_world_position_enu()[:2]
        self._log(f"Start world ENU: ({self._last_position[0]:.1f}, {self._last_position[1]:.1f})")
        
        coverage = 0.0
        
        try:
            for it in range(self.config.max_iterations):
                self.iteration = it + 1
                self._log(f"----- Iter {self.iteration} -----")
                
                # Scan
                self._scan_phase()
                
                # Build map
                pts_all = self.drone.load_all_points()
                if pts_all.size == 0:
                    continue
                
                if self._shared:
                    self._shared.update_points(self.drone.name, pts_all)
                
                try:
                    self.grid_info = self.grid_builder.build(pts_all)
                except ValueError:
                    continue
                
                # Compute coverage
                coverage = self._compute_region_coverage()
                global_cov = self._shared.get_global_coverage(self.config.cell_size) if self._shared else coverage
                
                self._log(f"Local: {coverage*100:.1f}% | Global: {global_cov*100:.1f}%")
                
                # Stale check
                if abs(coverage - self._last_coverage) < 0.005:
                    self._stale_count += 1
                else:
                    self._stale_count = 0
                self._last_coverage = coverage
                
                # Plan
                result = self._plan_phase()
                
                if self.config.save_debug_plots:
                    self._save_debug_plot(result)
                
                # Termination
                should_stop, reason = self._check_termination(coverage, global_cov, result)
                if should_stop:
                    self._log(f"STOPPING: {reason}")
                    break
                
                if not result.success:
                    self._log("No path, exploring nearby...")
                    self._explore_nearby()
                    continue
                
                # Move
                self._move_phase(result)
                
                # Track distance
                current = self.drone.get_world_position_enu()[:2]
                if self._last_position is not None:
                    dist = float(np.linalg.norm(current - self._last_position))
                    self._total_distance += dist
                self._last_position = current
        
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.drone.land()
            self.drone.disconnect()
            if self.logger:
                self.logger.close()
        
        self._log(f"========== DONE: {coverage*100:.1f}%, {self._total_distance:.0f}m ==========")
        return coverage

    def _check_termination(self, local_cov: float, global_cov: float, 
                           result: PlanResult) -> Tuple[bool, str]:
        if local_cov >= self.config.coverage_target:
            return True, f"Local target reached ({local_cov*100:.1f}%)"
        if global_cov >= self.config.coverage_target:
            return True, f"Global target reached ({global_cov*100:.1f}%)"
        if self._stale_count >= self.config.stale_iterations_limit:
            return True, f"Stale for {self._stale_count} iterations"
        return False, ""

    def _explore_nearby(self):
        """Move to random nearby spot when stuck."""
        pos = self.drone.get_position_ned()
        
        for _ in range(4):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(5, 15)
            
            new_x = pos[0] + dist * np.cos(angle)
            new_y = pos[1] + dist * np.sin(angle)
            
            try:
                self.drone.client.moveToPositionAsync(
                    float(new_x), float(new_y), -self.config.altitude,
                    self.config.speed, timeout_sec=10.0,
                    vehicle_name=self.drone.name
                ).join()
                
                for _ in range(5):
                    self.drone.scan_and_save()
                    time.sleep(0.1)
                return
            except:
                continue

    def _compute_region_coverage(self) -> float:
        """Compute coverage within assigned ENU region."""
        if self.grid_info is None:
            return 0.0
        
        region_mask = self._build_region_mask()
        if region_mask is None:
            # No region set - use full grid coverage
            return self.grid_info.coverage_ratio
        
        covered = int((self.grid_info.grid & region_mask).sum())
        total = int(region_mask.sum())
        
        if total == 0:
            self._log("WARNING: Region mask is empty!")
            return 0.0
        
        return covered / total

    def _build_region_mask(self) -> Optional[np.ndarray]:
        """Build mask for assigned region in ENU coordinates."""
        if self.grid_info is None:
            return None
        
        if self._region_enu is None:
            return None  # No region assigned, use full grid
        
        x_min, x_max, y_min, y_max = self._region_enu
        
        h, w = self.grid_info.grid.shape
        
        # Grid is in ENU - cell centers
        xs = self.grid_info.x_min + (np.arange(w) + 0.5) * self.grid_info.cell_size
        ys = self.grid_info.y_min + (np.arange(h) + 0.5) * self.grid_info.cell_size
        Xs, Ys = np.meshgrid(xs, ys)
        
        # Simple bounds check in ENU
        in_region = (
            (Xs >= x_min) & (Xs <= x_max) &
            (Ys >= y_min) & (Ys <= y_max)
        )
        
        region_cells = int(in_region.sum())
        if region_cells == 0:
            self._log(f"WARNING: No cells in region! Grid X=[{self.grid_info.x_min:.0f},{self.grid_info.x_min + w*self.grid_info.cell_size:.0f}] "
                     f"Y=[{self.grid_info.y_min:.0f},{self.grid_info.y_min + h*self.grid_info.cell_size:.0f}]")
        
        return in_region

    def _scan_phase(self):
        for _ in range(self.config.frames_per_scan):
            self.drone.scan_and_save()
            time.sleep(self.config.scan_interval)

    def _plan_phase(self) -> PlanResult:
        if self.grid_info is None:
            return PlanResult(None, None, False)
        
        # Use world ENU position
        pos_enu = self.drone.get_world_position_enu()[:2]
        
        obstacle_mask = self.planner.build_obstacle_mask(self.grid_info)
        obstacle_mask = self._carve_escape_bubble(obstacle_mask, pos_enu)
        
        deadzone = compute_deadzone_mask(
            self.grid_info, pos_enu, self.config.deadzone_radius
        )
        frontier_mask = self.planner.find_valid_frontiers(
            self.grid_info, obstacle_mask, deadzone
        )
        
        # Restrict frontiers to region (but don't add region as obstacle)
        region_mask = self._build_region_mask()
        if region_mask is not None:
            frontier_mask = frontier_mask & region_mask
        
        # Remove visited
        if self._visited_cells:
            ys, xs = np.where(frontier_mask)
            for y, x in zip(ys, xs):
                if (y, x) in self._visited_cells:
                    frontier_mask[y, x] = False
        
        n_frontiers = int(frontier_mask.sum())
        self._log(f"Frontiers: {n_frontiers}")
        
        if n_frontiers == 0:
            return PlanResult(None, None, False)
        
        result = self.planner.plan(
            self.grid_info, obstacle_mask, frontier_mask, pos_enu
        )
        
        if result.goal_cell:
            self._visited_cells.add(result.goal_cell)
        
        return result

    def _carve_escape_bubble(self, obstacle_mask: np.ndarray, 
                             pos_enu: np.ndarray) -> np.ndarray:
        if self.grid_info is None:
            return obstacle_mask
        
        bubble = compute_deadzone_mask(
            self.grid_info, pos_enu, self.config.escape_radius
        )
        result = obstacle_mask.copy()
        result[bubble] = False
        return result

    def _move_phase(self, plan: PlanResult) -> bool:
        if plan.path is None or len(plan.path) <= 1:
            return False
        
        step_idx = min(self.config.max_step_cells, len(plan.path) - 1)
        step_idx = max(1, step_idx)
        
        iy, ix = plan.path[step_idx]
        x_enu, y_enu = self.grid_info.cell_to_enu(ix, iy)
        
        # ENU to NED for AirSim commands
        # But we need to account for the drone's local NED vs world ENU
        # Simplest: move in local NED relative to current position
        
        current_world = self.drone.get_world_position_enu()
        target_world = np.array([x_enu, y_enu])
        delta_enu = target_world - current_world[:2]
        
        # Get current local NED
        current_ned = self.drone.get_position_ned()
        
        # Delta in local NED (swap X/Y for ENU->NED)
        target_ned_x = current_ned[0] + delta_enu[1]  # N = delta_Y_enu
        target_ned_y = current_ned[1] + delta_enu[0]  # E = delta_X_enu
        z_ned = -self.config.altitude
        
        dist = float(np.linalg.norm(delta_enu))
        self._log(f"Moving {dist:.1f}m")
        
        if dist < 1.0:
            return False
        
        try:
            self.drone.client.moveToPositionAsync(
                float(target_ned_x), float(target_ned_y), z_ned,
                self.config.speed,
                timeout_sec=self.config.move_timeout,
                vehicle_name=self.drone.name
            ).join()
        except Exception as e:
            self._log(f"Move error: {e}")
            return True
        
        return False

    def _save_debug_plot(self, plan: Optional[PlanResult] = None):
        if self.grid_info is None:
            return
        
        try:
            obstacle_mask = self.planner.build_obstacle_mask(self.grid_info)
            frontier_mask = self.planner.find_valid_frontiers(self.grid_info, obstacle_mask)
            
            region_mask = self._build_region_mask()
            if region_mask is not None:
                frontier_mask = frontier_mask & region_mask
            
            vis = np.zeros_like(self.grid_info.grid, dtype=np.uint8)
            vis[self.grid_info.grid] = 1
            vis[frontier_mask] = 2
            vis[self.grid_info.wall_mask] = 3
            vis[obstacle_mask] = 4
            
            # Draw region boundary correctly
            if region_mask is not None:
                # Only show boundary at edge of region
                from scipy import ndimage
                try:
                    boundary = region_mask & ~ndimage.binary_erosion(region_mask)
                except:
                    boundary = region_mask ^ dilate_mask(~region_mask, 1)
                vis[boundary] = 5
            
            cmap = ListedColormap([
                "#202020", "#55aa55", "#ff4444", 
                "#3366ff", "#ffcc00", "#ff00ff"
            ])
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(vis, origin="lower", cmap=cmap, interpolation="nearest")
            
            # Drone position in grid coords
            pos = self.drone.get_world_position_enu()[:2]
            ix, iy = self.grid_info.enu_to_cell(pos[0], pos[1])
            ax.scatter(ix, iy, s=100, marker="*", color="cyan", zorder=10)
            
            if plan and plan.path and len(plan.path) > 1:
                path_y = [p[0] for p in plan.path]
                path_x = [p[1] for p in plan.path]
                ax.plot(path_x, path_y, 'w-', linewidth=2, alpha=0.8)
                ax.scatter(path_x[-1], path_y[-1], s=80, marker="x", color="white", zorder=10)
            
            cov = self._compute_region_coverage()
            ax.set_title(f"{self.drone.name} - Iter {self.iteration} - {cov*100:.1f}%")
            
            fig.tight_layout()
            fig.savefig(self.debug_dir / f"coverage_{self.iteration:03d}.png", dpi=100)
            plt.close(fig)
        except Exception as e:
            self._log(f"Plot error: {e}")