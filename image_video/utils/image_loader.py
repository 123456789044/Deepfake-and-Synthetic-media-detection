import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_images(dataset_path):

    images = []
    labels = []

    for label_name, label in [("real",0),("fake",1)]:

        folder = os.path.join(dataset_path,label_name)

        for file in os.listdir(folder):

            img_path = os.path.join(folder,file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img = img/255.0

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)