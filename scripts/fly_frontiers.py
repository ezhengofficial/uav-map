import airsim
import time
import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fly a drone to coverage frontier waypoints.")
    parser.add_argument("--vehicle", type=str, default="Drone1",
                        help="Vehicle name in settings.json (default: Drone1).")
    parser.add_argument("--alt", type=float, default=-8.0,
                        help="Flight altitude in NED (negative is up, default: -8.0).")
    parser.add_argument("--speed", type=float, default=5.0,
                        help="Flight speed in m/s (default: 5.0).")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Seconds to hover at each waypoint (default: 1.0).")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    wp_path = script_path.parents[1] / "data" / "frontier_waypoints.npz"

    if not wp_path.exists():
        raise SystemExit(f"[ERROR] Frontier waypoints file not found: {wp_path}")

    data = np.load(wp_path)
    waypoints_ned = data["waypoints_ned"]  # shape (N, 2): [X_ned, Y_ned]

    if waypoints_ned.shape[0] == 0:
        print("[WARN] No frontier waypoints found; nothing to fly to.")
        return

    client = airsim.MultirotorClient()
    client.confirmConnection()

    name = args.vehicle
    alt = args.alt
    speed = args.speed

    print(f"[INFO] Enabling API control and arming {name}...")
    client.enableApiControl(True, vehicle_name=name)
    client.armDisarm(True, vehicle_name=name)

    print(f"[INFO] Taking off {name}...")
    client.takeoffAsync(vehicle_name=name).join()

    # Move to initial altitude
    client.moveToZAsync(alt, speed, vehicle_name=name).join()

    print(f"[INFO] Flying {name} to {len(waypoints_ned)} frontier waypoints...")
    for i, (X_ned, Y_ned) in enumerate(waypoints_ned):
        print(f"  [{i+1}/{len(waypoints_ned)}] -> (X={X_ned:.2f}, Y={Y_ned:.2f}, Z={alt:.2f})")
        client.moveToPositionAsync(float(X_ned), float(Y_ned), float(alt),
                                   speed, vehicle_name=name).join()
        time.sleep(args.pause)

    print(f"[INFO] Returning {name} to start altitude (optional: you can skip this).")
    # You could add a return-to-home here if desired.

    print(f"[INFO] Landing {name}...")
    client.landAsync(vehicle_name=name).join()
    client.armDisarm(False, vehicle_name=name)
    client.enableApiControl(False, vehicle_name=name)
    print("[DONE] Frontier flight complete.")

if __name__ == "__main__":
    main()
