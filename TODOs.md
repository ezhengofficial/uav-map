# TODOs

## Project Related Stuff
* Add flask, keyboard to `requirements.txt`
* For `lidar/app.py`
    * Update path of the merge files to be read from `uav-map/data/` instead of `uav-map/data/lidar/data`
    * Implement logic to handle if directory d.n.e. 
        * Solution 1: Have a dummy file
        * Solution 2: IDK 
* Add/Integrate the following stuff from `uav-map/scripts`
    * `merge.py` -> `/lidar`
        * We should try seeing what happens if we merge all from .las files from the drones first. If that works we should just do this instead
    * `autonomous_mapper.py`
        * Absolutely no clue how to integrate this... figure out later...
* Write scripts for
    * Writing a working `settings.json` file
        * Main idea is to set-up the number of drones in `settings.json`
            * Lidar
            * Drone Name
            * Position
    * write a `.bat` for the whole lidar thing
        * Goal: Run in the background and continuously merges and
        * Side Note: We may reduce the complexity of `.bat` to simply running `app.py`
            * `lasviewer` has been reconfigured to continuously update the webpage every few seconds. We just need to add a few lines of logic handling merging.

* Implement logging correctly
    * CLean up some of the print
    * Clean up console.log in `lidar/static/*.js`


## Part 2: 
* The fun part.... Update the environment to a more realistic one. 

## Part 3:
* Presentation Slide Deck
    * Keep it short and have it as a pdf -> Attach to github
* README.md
