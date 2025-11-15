import keyboard
from drone_manager import DroneManager

def main():
    manager = DroneManager()

    print("Keyboard controls: t=takeoff, l=land, s=stop, n=navigate square, q=quit, w =start lidar, e=stop lidar")
    while True:
        if keyboard.is_pressed("t"):
            manager.takeoff_all()
        elif keyboard.is_pressed("l"):
            manager.land_all()
        elif keyboard.is_pressed("s"):
            manager.stop_all()
        elif keyboard.is_pressed("w"):
            manager.start_lidar_logging()
        elif keyboard.is_pressed("e"):
            manager.stop_lidar_logging()
        # elif keyboard.is_pressed("n"):
        #     manager.fly_square_all()
        elif keyboard.is_pressed("q"):
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
