import os
import cv2
from tqdm import tqdm

DATASET_PATH = "../dataset"
OUTPUT_PATH = "../frames_dataset"

IMG_SIZE = 128
FRAME_SKIP = 10

for label in ["Real", "Fake"]:

    video_folder = os.path.join(DATASET_PATH, label)

    save_folder = os.path.join(
        OUTPUT_PATH,
        label.lower()
    )

    os.makedirs(save_folder, exist_ok=True)

    for video in tqdm(os.listdir(video_folder)):

        video_path = os.path.join(video_folder, video)

        cap = cv2.VideoCapture(video_path)

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:

                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

                frame_name = f"{video}_{frame_id}.jpg"

                cv2.imwrite(
                    os.path.join(save_folder, frame_name),
                    frame
                )

            frame_id += 1

        cap.release()