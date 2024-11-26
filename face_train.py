import cv2
import numpy as np
from PIL import Image
import os

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []
    for image_path in image_paths:
        img = np.array(Image.open(image_path).convert('L'), 'uint8')
        id = int(os.path.split(image_path)[-1].split("-")[1])
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids

if __name__ == "__main__":
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n[INFO] Training...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
