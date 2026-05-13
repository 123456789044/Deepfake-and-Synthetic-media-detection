import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_audio_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# -------------------------
# Load Model
# -------------------------
print("Loading audio model...")
model = load_model(MODEL_PATH, compile=False)
print("Audio model loaded successfully")

# -------------------------
# Load Scaler
# -------------------------
print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully")


# -------------------------
# Feature Extraction
# -------------------------
def extract_features(file_path):

    try:

        print("Loading audio:", file_path)

        audio, sr = librosa.load(file_path, sr=22050)

        # normalize audio
        audio = librosa.util.normalize(audio)

        # fixed length (3 seconds)
        target_length = 3 * 22050

        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # MFCC extraction
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        # average MFCC
        mfcc = np.mean(mfcc.T, axis=0)

        print("MFCC shape:", mfcc.shape)

        return mfcc

    except Exception as e:

        print("Feature extraction error:", e)
        return None


# -------------------------
# Prediction Function
# -------------------------
def predict_audio(file_path):

    try:

        print("\nProcessing audio file:", file_path)

        features = extract_features(file_path)

        if features is None:
            return "Error processing audio"

        # reshape for model
        features = features.reshape(1, -1)

        # apply scaler from training
        features = scaler.transform(features)

        print("Input to model:", features.shape)

        # prediction
        prediction = model.predict(features)

        score = float(prediction[0][0])

        print("Prediction score:", score)

        confidence = round(score * 100, 2)

        if score < 0.5:
            return f"Fake Audio 🚨 ({100-confidence}% confidence)"
        else:
            return f"Real Audio ✅ ({confidence}% confidence)"

    except Exception as e:

        print("Audio Prediction Error:", e)

        return "Error detecting audio"