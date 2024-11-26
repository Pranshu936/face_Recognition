import numpy as np
import json
import cv2
import os

def create_directory(directory: str):
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

def get_face_id(directory: str) -> int:
    user_ids = [int(os.path.split(f)[-1].split("-")[1]) for f in os.listdir(directory)]
    user_ids = sorted(set(user_ids))
    max_user_ids = 1 if not user_ids else max(user_ids) + 1
    for i in range(max_user_ids):
        if i not in user_ids:
            return i
    return max_user_ids

def save_name(face_id: int, face_name: str, filename: str):
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            names_json = json.load(f)
    names_json[face_id] = face_name
    with open(filename, 'w') as f:
        json.dump(names_json, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = 'images'
    cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
    names_json_filename = 'names.json'

    create_directory(directory)
    faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    face_name = input('Enter user name: ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)
    
    print('Initializing face capture. Look at the camera and wait...')
    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f'./images/Users-{face_id}-{count}.jpg', gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        if cv2.waitKey(100) & 0xFF < 30 or count >= 30:
            break

    print('Success! Exiting Program.')
    cam.release()
    cv2.destroyAllWindows()
