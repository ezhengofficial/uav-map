from flask import Flask, send_file

app = Flask(__name__, static_folder="static")
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
    response = send_file("data/merged.las")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == "__main__":
    app.run(port=8000)
