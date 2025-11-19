import airsim
import time

# List of vehicle names as defined in settings.json
VEHICLES = ["Drone1", "Drone2", "Drone3"]

# Simple target positions for each drone (AirSim NED frame)
# (x, y, z) with z negative = up
WAYPOINTS = {
    "Drone1": (  0,  0, -5),
    "Drone2": ( 10,  0, -7),   # slightly higher
    "Drone3": (  0, 10, -9)    # even higher
}

SPEED = 3.0
HOVER_TIME = 5.0  # seconds to hover at waypoint


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Take control & arm all drones
    for name in VEHICLES:
        print(f"[INFO] Enabling API control and arming {name}...")
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    # Take off all drones
    print("[INFO] Taking off all drones...")
    takeoff_futures = [client.takeoffAsync(vehicle_name=name) for name in VEHICLES]
    for f in takeoff_futures:
        f.join()

    # Move each drone to its own waypoint
    print("[INFO] Moving drones to waypoints...")
    move_futures = []
    for name in VEHICLES:
        wp = WAYPOINTS.get(name, (0, 0, -5))
        x, y, z = wp
        print(f"  - {name} -> ({x}, {y}, {z})")
        move_futures.append(
            client.moveToPositionAsync(x, y, z, SPEED, vehicle_name=name)
        )
    for f in move_futures:
        f.join()

    print(f"[INFO] Hovering for {HOVER_TIME} seconds...")
    time.sleep(HOVER_TIME)

    # Land all drones
    print("[INFO] Landing all drones...")
    land_futures = [client.landAsync(vehicle_name=name) for name in VEHICLES]
    for f in land_futures:
        f.join()

    # Disarm and release control
    for name in VEHICLES:
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
        print(f"[INFO] {name} disarmed and API control released.")

    print("[DONE] Multi-drone hover test complete.")


if __name__ == "__main__":
    main()
