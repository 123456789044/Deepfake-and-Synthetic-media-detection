from flask import Flask, render_template, request, jsonify
import os

from audio.predict import predict_audio
from image_video.predictor import predict_image_video

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/audio")
def audio_page():
    return render_template("audio.html")


@app.route("/image")
def image_page():
    return render_template("image.html")


@app.route("/video")
def video_page():
    return render_template("video.html")


# -------------------------
# Audio Prediction
# -------------------------
@app.route("/predict_audio", methods=["POST"])
def predict_audio_route():

    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"result": "No file selected"})

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    prediction = predict_audio(filepath)

    return jsonify({"result": prediction})


# -------------------------
# Image / Video Prediction
# -------------------------
@app.route("/predict_media", methods=["POST"])
def predict_media_route():

    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"result": "No file selected"})

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    prediction = predict_image_video(filepath)

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)