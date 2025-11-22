import sys
import time
import argparse
from drone_manager import DroneManager
from exploration.explorer import ExplorationConfig


def run_interactive(manager: DroneManager):
    """
    Interactive keyboard control mode.
    Uses simple input() for cross-platform compatibility.
    """
    print("\n" + "="*50)
    print("Drone Control Menu")
    print("="*50)
    print("  t - Takeoff all")
    print("  l - Land all")
    print("  s - Stop/hover all")
    print("  w - Start LiDAR logging")
    print("  e - Stop LiDAR logging")
    print("  x - Start parallel exploration (all drones)")
    print("  p - Show exploration status")
    print("  q - Quit")
    print("="*50 + "\n")
    
    try:
        while True:
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == 't':
                manager.takeoff_all()
            elif cmd == 'l':
                manager.land_all()
            elif cmd == 's':
                manager.stop_all()
            elif cmd == 'w':
                manager.start_lidar_logging(interval=0.5)
            elif cmd == 'e':
                manager.stop_lidar_logging()
            elif cmd == 'x':
                print("Starting parallel exploration...")
                print("Available strategies: vertical, horizontal, grid")
                strategy = input("Partition strategy [vertical]: ").strip() or "vertical"
                manager.start_exploration(partition_strategy=strategy)
            elif cmd == 'p':
                status = manager.get_exploration_status()
                print("Exploration status:")
                for name, active in status.items():
                    state = "ACTIVE" if active else "FINISHED"
                    print(f"  {name}: {state}")
            elif cmd == 'q':
                print("Exiting...")
                break
            else:
                print(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        manager.shutdown()


def run_exploration(manager: DroneManager, config: ExplorationConfig, 
                    strategy: str = "vertical"):
    """Run autonomous parallel exploration directly."""
    print("[Main] Starting parallel autonomous exploration...")
    print(f"[Main] Drones: {list(manager.drones.keys())}")
    print(f"[Main] Partition strategy: {strategy}")
    
    try:
        manager.start_exploration(config=config, partition_strategy=strategy)
        
        # Monitor progress
        while True:
            time.sleep(5.0)
            status = manager.get_exploration_status()
            active = [name for name, running in status.items() if running]
            
            if not active:
                print("[Main] All drones finished exploration")
                break
            
            print(f"[Main] Active explorers: {active}")
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted - shutting down...")
    finally:
        manager.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-drone parallel exploration system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python main.py
  
  # Direct exploration with vertical partitioning
  python main.py --mode explore --strategy vertical
  
  # Grid partitioning with custom bounds
  python main.py --mode explore --strategy grid --bounds -100 100 -100 100
        """
    )
    parser.add_argument("--mode", choices=["interactive", "explore"], 
                        default="interactive",
                        help="Run mode: interactive menu or auto-explore")
    parser.add_argument("--settings", type=str, default=None,
                        help="Path to AirSim settings.json")
    parser.add_argument("--altitude", type=float, default=5.0,
                        help="Flight altitude in meters (default: 5.0)")
    parser.add_argument("--coverage", type=float, default=0.85,
                        help="Target coverage ratio (default: 0.85)")
    parser.add_argument("--max-iters", type=int, default=100,
                        help="Maximum exploration iterations (default: 100)")
    parser.add_argument("--strategy", choices=["vertical", "horizontal", "grid"],
                        default="vertical",
                        help="Map partitioning strategy (default: vertical)")
    parser.add_argument("--bounds", type=float, nargs=4, 
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        default=[-150, 150, -150, 150],
                        help="World bounds in NED (default: -150 150 -150 150)")
    args = parser.parse_args()
    
    # Initialize manager
    try:
        manager = DroneManager(settings_file=args.settings)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Set world bounds
    manager.set_world_bounds(*args.bounds)
    
    # Build config
    config = ExplorationConfig()
    config.altitude = args.altitude
    config.coverage_target = args.coverage
    config.max_iterations = args.max_iters
    
    # Run selected mode
    if args.mode == "interactive":
        run_interactive(manager)
    else:
        run_exploration(manager, config, args.strategy)


if __name__ == "__main__":
    main()