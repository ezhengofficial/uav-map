# AirSim Lidar Map Project
Drone Mapping with AirSim

## About the project
This project leverages Unreal Engine and Microsoft AirSim to simulate drones performing autonomous mapping missions. Multiple drones fly across a virtual environment and coordinate to cover the frontier waypoint and collect LiDAR point cloud data in real-time. The system merges these data streams into one LAS file, enabling visualization of the mapped area.

### Problem Statment

Search and Rescue missions often take place in hazardous and unpredictable environments with limited time and resources. This project leverages UAVS to help solve some of the hardships associated with search and rescue missions. 
1. UAVs offer additional mobility by reaching difficult areas quickly, which is crucial in time-sensitive situations.
2. UAVs aren't limited resources, as swarms of UAVs can be set up for greater mobility and coverage area.
3. UAVs can provide real-time data for clients and can be equipped with different sensors.
This project aims to simulate multiple drones mapping an environment and providing real-time data for users. This project focuses specifically on using LiDAR to map surrounding areas.

### Solution

We addressed this problem by
* Implemented multiple drones to cover a larger area of the map in less time
* Equipped drones with LiDAR to generate a map of the surrounding
* Implemented real-time footage of LiDAR Map

<!-- TODO - Figure out what works later-->


## Getting Started
Instructions to help set up the project locally and run it


### Prerequisites

* See [Unreal Engine](https://www.unrealengine.com/en-US/download)
* Download [Blocks](https://github.com/microsoft/AirSim/releases/download/v1.8.0-linux/Blocks.zip) 
    * See [here](https://github.com/microsoft/AirSim/releases/) for more information

### Installations

1. Clone the repository
    ```sh
    git clone https://github.com/ezhengofficial/uav-map.git
    cd uav-map
    ```

2. Install dependencies
    ```sh
    pip install -r requirements.txt
    ```

3. Setup `settings.json`
    * An example of `settings.json` can be found [here](config/settings.json)

4. Launch AirSim Simulator
    * `Blocks.exe` for this project

5. Run code to fly drone and collect data
    ```sh
    cd scripts
    python multi_realtime_map.py
    ```
6. Run code to visualize LiDAR data in a separate terminal
    ```sh
    cd lidar
    python app.py
    ```
7. Open local host
    * `http://127.0.0.1:8000`

## Bugs 

* If the code does not connect to the simulator, check that
    * `settings.json` and the simulator established connection
    * Restart your device
* After a few million LiDAR data points, there will be some lag in updating LasView

### Work In Progress 

* Currently, the modularity of our code is a work in progress.  Besides drones not coordinating well to map the environment, everything else works
    * To see some results for the modular aspect, run 
    ```sh
    cd drone
    python main.py --setings [PATH_TO_SETTINGS_JSON_FILE]
    ```
    Followed by the following input
    ```
    x
    horizontal
    ```
    instead of `python multi_realtime_map.py`



## License

See [License](LICENSE) for license information

## Acknowledgements
* [LasViewer](https://lasviewer.github.io/)
* [AirSim](https://microsoft.github.io/AirSim/)