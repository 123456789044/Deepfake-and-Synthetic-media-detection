import os
import cv2
from mtcnn import MTCNN

INPUT_DIR = "dataset"
OUTPUT_DIR = "faces_dataset"

IMG_SIZE = 224
FRAME_SKIP = 10

detector = MTCNN()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in ["real", "fake"]:

    input_path = os.path.join(INPUT_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, label)

    os.makedirs(output_path, exist_ok=True)

    for video_name in os.listdir(input_path):

        video_path = os.path.join(input_path, video_name)

        cap = cv2.VideoCapture(video_path)

        frame_id = 0
        saved_faces = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = detector.detect_faces(rgb)

            for face in faces:

                x, y, w, h = face['box']

                x = max(0, x)
                y = max(0, y)

                face_crop = frame[y:y+h, x:x+w]

                if face_crop.size == 0:
                    continue

                face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

                save_name = f"{video_name}_{frame_id}.jpg"

                save_path = os.path.join(output_path, save_name)

                cv2.imwrite(save_path, face_crop)

                saved_faces += 1

            frame_id += 1

        cap.release()

        print(f"{video_name} → {saved_faces} faces saved")

print("Face extraction completed.")