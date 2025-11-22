from flask import Flask, send_file
import merge
<<<<<<< HEAD
from files_path import DATA_DIR


app = Flask(__name__, static_folder="static")
MERGED_LAS_FILE = DATA_DIR / "final/merged.las"
=======

app = Flask(__name__, static_folder="static")
>>>>>>> origin/fix_lidar
FINAL_FILE = "data/merged.las" # TODO Update this

@app.route("/")
def index():
    return app.send_static_file("index.html")

# @app.route("/<path:filename>")
# def static_files(filename):
#     return app.send_static_file(filename)

@app.route("/index.data")
def wasm_data():
    return app.send_static_file("index.data")


@app.route("/merged.las")
def merged_las():
    # TODO - Merge files beforehand
    merge.merge_data()
<<<<<<< HEAD
    response = send_file(MERGED_LAS_FILE)
=======
    response = send_file("data/merged.las")
>>>>>>> origin/fix_lidar
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == "__main__":
    app.run(port=8000)
