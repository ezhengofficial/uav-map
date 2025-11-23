from flask import Flask, send_file, jsonify, request
import merge
from files_path import MERGED_LAS_FILE


app = Flask(__name__, static_folder="static")

CURRENT_VIEW_OPTIONS = {}

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
    response = send_file(MERGED_LAS_FILE)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/view-options", methods=["GET", "POST"])
def view_options():
    global CURRENT_VIEW_OPTIONS
    if request.method == "POST":
        CURRENT_VIEW_OPTIONS = request.json or {}
        return jsonify({"status": "saved"})
    return jsonify(CURRENT_VIEW_OPTIONS)

if __name__ == "__main__":
    app.run(port=8000)
