# AirSim Lidar Map Project
Drone Mapping with AirSim

## About the project
This project leverages Unreal Engine and Microsoft Airsim to simulate  drones performing autonomous mapping missions. Multiple drone fly across virutal environment and coordinate to cover frontier waypoint and collect
LiDAR point cloud data in real-time. The system merge these data streams into one LAS file enabling visualization of the mapped area.

### Problem Statment

Search and Rescue missions often take place in hazardous and unpredictable environments with limited time and resource. UAVs offer additional mobility by reaching difficult areas more quickly and gathering surrounding data. This project aims to simulate multiple drones mapping an environment and providing real-time data for users. This project focuses specifcally on using LiDAR to map surrounding areas

### Solution

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
6. Run code to visualize LiDAR data in a seperate terminal
    ```sh
    cd lidar
    python app.py
    ```

<!-- TODO - Run the code via batch? -->

## Bugs

* If the code does not connect to the simulator, check that
    * `settings.json` and the simulator established connection
    * Restart your device

## License

See [License](LICENSE) for license information

## Acknowledgements
* [LasViewer](https://lasviewer.github.io/)
* [AirSim](https://microsoft.github.io/AirSim/)