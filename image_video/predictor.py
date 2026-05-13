import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 224
FRAME_SKIP = 5
MIN_FACE_SIZE = 40   # prevent tiny faces crashing model


# -----------------------------
# Fix for older EfficientNet models
# -----------------------------
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# -----------------------------
# Get project base directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "deepfake_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print("Loading deepfake model...")

model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
)

print("Model loaded successfully")

# Initialize face detector
detector = MTCNN()


# -----------------------------
# Preprocess face
# -----------------------------
def preprocess_face(face):

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    return face


# -----------------------------
# Predict Image
# -----------------------------
def predict_image(image_path):

    try:

        print("Processing image:", image_path)

        image = cv2.imread(image_path)

        if image is None:
            return {"result": "Invalid image file", "confidence": 0}

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb)

        if len(faces) == 0:
            return {"result": "No face detected", "confidence": 0}

        scores = []

        for face_data in faces:

            x, y, w, h = face_data["box"]

            x = max(0, x)
            y = max(0, y)

            # skip tiny faces
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            face = image[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = preprocess_face(face)

            pred = model.predict(face, verbose=0)[0][0]

            scores.append(pred)

        if len(scores) == 0:
            return {"result": "Face processing failed", "confidence": 0}

        avg_score = np.mean(scores)

        confidence = max(avg_score, 1 - avg_score)

        if avg_score < 0.5:
            result = "Fake Image 🚨"
        else:
            result = "Real Image ✅"

        return {
            "result": result,
            "confidence": float(confidence * 100)
        }

    except Exception as e:

        print("IMAGE ERROR:", e)

        return {
            "result": "Error processing image",
            "confidence": 0
        }


# -----------------------------
# Predict Video
# -----------------------------
def predict_video(video_path):

    try:

        print("Processing video:", video_path)

        cap = cv2.VideoCapture(video_path)

        frame_id = 0
        scores = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # skip frames for speed
            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = detector.detect_faces(rgb)

            # skip frame if no face
            if len(faces) == 0:
                frame_id += 1
                continue

            for face_data in faces:

                x, y, w, h = face_data["box"]

                x = max(0, x)
                y = max(0, y)

                # skip tiny faces
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue

                face = frame[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                face = preprocess_face(face)

                pred = model.predict(face, verbose=0)[0][0]

                scores.append(pred)

            frame_id += 1

        cap.release()

        if len(scores) == 0:
            return {"result": "No face detected in video", "confidence": 0}

        avg_score = np.mean(scores)

        confidence = max(avg_score, 1 - avg_score)

        if avg_score < 0.5:
            result = "Fake Video 🚨"
        else:
            result = "Real Video ✅"

        return {
            "result": result,
            "confidence": float(confidence * 100)
        }

    except Exception as e:

        print("VIDEO ERROR:", e)

        return {
            "result": "Error processing video",
            "confidence": 0
        }


# -----------------------------
# Router
# -----------------------------
def predict_image_video(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    image_ext = [".jpg", ".jpeg", ".png"]
    video_ext = [".mp4", ".avi", ".mov", ".mkv"]

    if ext in image_ext:
        return predict_image(file_path)

    elif ext in video_ext:
        return predict_video(file_path)

    else:
        return {
            "result": "Unsupported file format",
            "confidence": 0
        }