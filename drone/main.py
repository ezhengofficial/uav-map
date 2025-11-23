"""
Main entry point with separate logging per drone.
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from drone_manager import DroneManager
from exploration.explorer import ExplorationConfig
from files_path import DATA_DIR


def setup_logging_dirs():
    """Create debug directory for this run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path(DATA_DIR) / "debug" / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def run_interactive(manager: DroneManager):
    """Interactive mode."""
    print("\n" + "="*50)
    print("Drone Control")
    print("="*50)
    print("  t - Takeoff")
    print("  l - Land")
    print("  s - Stop/hover")
    print("  c - Calibrate GPS")
    print("  x - Start exploration")
    print("  p - Status")
    print("  q - Quit")
    print("="*50 + "\n")
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == 't':
                manager.takeoff_all()
            elif cmd == 'l':
                manager.land_all()
            elif cmd == 's':
                manager.stop_all()
            elif cmd == 'c':
                manager.calibrate_gps_origin()
            elif cmd == 'x':
                strategy = input("Strategy [quadrant]: ").strip() or "quadrant"
                manager.start_exploration(partition_strategy=strategy)
            elif cmd == 'p':
                for name, active in manager.get_exploration_status().items():
                    print(f"  {name}: {'ACTIVE' if active else 'DONE'}")
            elif cmd == 'q':
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        manager.shutdown()


def run_exploration(manager: DroneManager, config: ExplorationConfig, 
                    strategy: str = "quadrant"):
    """Run parallel exploration."""
    print(f"[Main] Starting exploration with {len(manager.drones)} drones")
    print(f"[Main] Strategy: {strategy}")
    print(f"[Main] Logs will be saved to each drone's debug folder")
    
    try:
        manager.start_exploration(config=config, partition_strategy=strategy)
        
        while True:
            time.sleep(5.0)
            status = manager.get_exploration_status()
            active = [n for n, running in status.items() if running]
            
            if not active:
                print("[Main] All drones finished")
                break
            
            print(f"[Main] Active: {active}")
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
    finally:
        manager.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Multi-drone exploration")
    parser.add_argument("--mode", choices=["interactive", "explore"], 
                        default="interactive")
    parser.add_argument("--settings", type=str, default=None)
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--strategy", choices=["vertical", "horizontal", "grid", "quadrant"],
                        default="quadrant")
    parser.add_argument("--bounds", type=float, nargs=4, 
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        default=[-100, 100, -100, 100],
                        help="World bounds in ENU meters")
    args = parser.parse_args()
    
    # Initialize
    try:
        manager = DroneManager(settings_file=args.settings)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Set world bounds in ENU
    manager.set_world_bounds_enu(*args.bounds)
    
    # Config
    config = ExplorationConfig()
    config.altitude = args.altitude
    config.coverage_target = args.coverage
    config.max_iterations = args.max_iters
    config.log_to_file = True  # Enable file logging
    
    if args.mode == "interactive":
        run_interactive(manager)
    else:
        run_exploration(manager, config, args.strategy)


if __name__ == "__main__":
    main()