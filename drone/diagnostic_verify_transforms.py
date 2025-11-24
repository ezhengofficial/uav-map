"""
Usage:
    python diagnostic_verify_transforms.py
"""
import time
import numpy as np
import airsim
from pathlib import Path
from drone_manager import DroneManager
from files_path import DATA_DIR


def test_single_drone_alignment(manager: DroneManager, drone_name: str):
    """Test that a single drone's scans align properly."""
    print(f"\n{'='*60}")
    print(f"Testing {drone_name}")
    print(f"{'='*60}")
    
    drone = manager.drones[drone_name]
    drone.connect()
    
    # Get initial GPS and set as origin for single-drone test
    state = drone.client.getMultirotorState(vehicle_name=drone_name)
    gps = state.gps_location
    print(f"Initial GPS: lat={gps.latitude:.8f}, lon={gps.longitude:.8f}, alt={gps.altitude:.2f}")
    
    # CRITICAL: Set GPS origin for this test
    drone.set_gps_origin(gps)
    print(f"Set GPS origin for single-drone test")
    
    # Takeoff
    print("Taking off...")
    drone.takeoff(altitude=5.0)
    time.sleep(2.0)
    
    # Get position in different coordinate systems
    pos_ned = drone.get_position_ned()
    pos_world_enu = drone.get_world_position_enu()
    
    # Check GPS
    state = drone.client.getMultirotorState(vehicle_name=drone_name)
    current_gps = state.gps_location
    
    print(f"\nDrone position:")
    print(f"  NED (local):  [{pos_ned[0]:7.2f}, {pos_ned[1]:7.2f}, {pos_ned[2]:7.2f}]")
    print(f"  ENU (world):  [{pos_world_enu[0]:7.2f}, {pos_world_enu[1]:7.2f}, {pos_world_enu[2]:7.2f}]")
    print(f"  GPS current:  lat={current_gps.latitude:.8f}, lon={current_gps.longitude:.8f}, alt={current_gps.altitude:.2f}")
    
    if drone._gps_origin:
        d_lat = current_gps.latitude - drone._gps_origin.latitude
        d_lon = current_gps.longitude - drone._gps_origin.longitude
        d_alt = current_gps.altitude - drone._gps_origin.altitude
        print(f"  GPS delta:    Δlat={d_lat:.8f}°, Δlon={d_lon:.8f}°, Δalt={d_alt:.2f}m")
    
    # Get orientation
    state = drone.client.getMultirotorState(vehicle_name=drone_name)
    q = state.kinematics_estimated.orientation
    print(f"  Quaternion:   [{q.w_val:.3f}, {q.x_val:.3f}, {q.y_val:.3f}, {q.z_val:.3f}]")
    
    # Capture LiDAR at 3 different positions
    print(f"\nCapturing scans at different positions...")
    positions = []
    
    for i in range(3):
        # Move to new position
        offset = i * 10.0
        target_ned = (pos_ned[0] + offset, pos_ned[1], -5.0)
        
        print(f"\n  Position {i+1}: Moving to NED {target_ned}...")
        drone.client.moveToPositionAsync(
            float(target_ned[0]), float(target_ned[1]), float(target_ned[2]),
            velocity=2.0, timeout_sec=15.0, vehicle_name=drone_name
        ).join()
        
        time.sleep(1.0)
        
        # Get actual position
        actual_ned = drone.get_position_ned()
        actual_world = drone.get_world_position_enu()
        
        # Get GPS
        state_check = drone.client.getMultirotorState(vehicle_name=drone_name)
        gps_check = state_check.gps_location
        
        print(f"    Actual NED:   [{actual_ned[0]:7.2f}, {actual_ned[1]:7.2f}, {actual_ned[2]:7.2f}]")
        print(f"    Actual World: [{actual_world[0]:7.2f}, {actual_world[1]:7.2f}, {actual_world[2]:7.2f}]")
        print(f"    GPS:          lat={gps_check.latitude:.8f}, lon={gps_check.longitude:.8f}")
        
        # Capture scan
        pts = drone.get_lidar_world_enu()
        if pts is not None and len(pts) > 0:
            print(f"    LiDAR points: {len(pts):,}")
            print(f"    Point range X: [{pts[:,0].min():7.2f}, {pts[:,0].max():7.2f}]")
            print(f"    Point range Y: [{pts[:,1].min():7.2f}, {pts[:,1].max():7.2f}]")
            print(f"    Point range Z: [{pts[:,2].min():7.2f}, {pts[:,2].max():7.2f}]")
            
            # Save for analysis
            drone.scan_and_save(colorize=True)
            positions.append((actual_world, pts))
        else:
            print(f"    WARNING: No LiDAR data!")
    
    # Check consistency
    if len(positions) >= 2:
        print(f"\n{'='*60}")
        print("Checking consistency between scans...")
        print(f"{'='*60}")
        
        for i in range(len(positions) - 1):
            pos1, pts1 = positions[i]
            pos2, pts2 = positions[i+1]
            
            drone_dist = np.linalg.norm(pos2 - pos1)
            
            # Expected: LiDAR data should move WITH the drone
            # If properly aligned, scans should show overlapping areas
            print(f"\nPositions {i+1} -> {i+2}:")
            print(f"  Drone moved: {drone_dist:.2f}m")
            print(f"  Scan {i+1}: {len(pts1):,} points")
            print(f"  Scan {i+2}: {len(pts2):,} points")
            
            # Check if there's reasonable overlap
            # (this is a rough check - actual overlap depends on scene)
            pts1_center = np.mean(pts1, axis=0)
            pts2_center = np.mean(pts2, axis=0)
            center_shift = np.linalg.norm(pts2_center - pts1_center)
            
            print(f"  Point cloud center shift: {center_shift:.2f}m")
            
            if abs(center_shift - drone_dist) > 5.0:
                print(f"  ⚠️  WARNING: Center shift doesn't match drone movement!")
                print(f"     Expected ~{drone_dist:.2f}m, got {center_shift:.2f}m")
            else:
                print(f"  ✓ Point cloud shift matches drone movement")
    
    # Land
    print(f"\nLanding...")
    drone.land()
    drone.disconnect()
    
    print(f"\n{'='*60}")
    print(f"Test complete for {drone_name}")
    print(f"Data saved to: {drone.data_dir}")
    print(f"{'='*60}\n")


def test_multi_drone_alignment(manager: DroneManager):
    """Test that multiple drones produce aligned data."""
    print(f"\n{'='*60}")
    print(f"Testing Multi-Drone Alignment")
    print(f"{'='*60}")
    
    if len(manager.drones) < 2:
        print("Need at least 2 drones for multi-drone test")
        return
    
    # Calibrate GPS origin
    print("\nCalibrating GPS origin...")
    manager.calibrate_gps_origin()
    time.sleep(1.0)
    
    # Takeoff all drones
    print("\nTaking off all drones...")
    manager.takeoff_all(altitude=5.0)
    time.sleep(2.0)
    
    # Get positions of all drones in world ENU
    print("\nDrone positions in world ENU frame:")
    for name, drone in manager.drones.items():
        pos = drone.get_world_position_enu()
        print(f"  {name}: [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]")
    
    # Capture simultaneous scans
    print("\nCapturing simultaneous scans...")
    manager.client.simPause(True)
    
    scan_data = {}
    for name, drone in manager.drones.items():
        pts = drone.get_lidar_world_enu()
        if pts is not None and len(pts) > 0:
            scan_data[name] = pts
            print(f"  {name}: {len(pts):,} points")
    
    manager.client.simPause(False)
    
    # Check if scans overlap or are properly separated
    if len(scan_data) >= 2:
        print("\nChecking spatial relationship between drone scans...")
        names = list(scan_data.keys())
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name1, name2 = names[i], names[j]
                pts1, pts2 = scan_data[name1], scan_data[name2]
                
                center1 = np.mean(pts1, axis=0)
                center2 = np.mean(pts2, axis=0)
                center_dist = np.linalg.norm(center2 - center1)
                
                pos1 = manager.drones[name1].get_world_position_enu()
                pos2 = manager.drones[name2].get_world_position_enu()
                drone_dist = np.linalg.norm(pos2 - pos1)
                
                print(f"\n  {name1} <-> {name2}:")
                print(f"    Drone separation: {drone_dist:.2f}m")
                print(f"    Scan center separation: {center_dist:.2f}m")
                
                if abs(center_dist - drone_dist) > 10.0:
                    print(f"    ⚠️  WARNING: Scan centers don't match drone positions!")
                else:
                    print(f"    ✓ Scans appear properly aligned")
    
    # Land all
    print("\nLanding all drones...")
    manager.land_all()
    
    print(f"\n{'='*60}")
    print("Multi-drone test complete")
    print(f"{'='*60}\n")


def main():
    print("\n" + "="*60)
    print("LiDAR Coordinate Transform Diagnostic Tool")
    print("="*60)
    
    try:
        manager = DroneManager()
    except Exception as e:
        print(f"ERROR: Failed to initialize manager: {e}")
        return
    
    print(f"\nFound {len(manager.drones)} drone(s)")
    for name in manager.drones.keys():
        print(f"  - {name}")
    
    # Test single drone first
    if len(manager.drones) > 0:
        first_drone = list(manager.drones.keys())[0]
        test_single_drone_alignment(manager, first_drone)
    
    # Test multi-drone if available
    if len(manager.drones) >= 2:
        test_multi_drone_alignment(manager)
    
    print("\n" + "="*60)
    print("Diagnostic complete!")
    print("\nNext steps:")
    print("1. Check the LAS files in data/<drone_name>/")
    print("2. Open in CloudCompare or similar viewer")
    print("3. Verify that:")
    print("   - Scans from same drone align properly")
    print("   - Scans from different drones share same coordinate system")
    print("   - No unexpected rotations or mirroring")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()