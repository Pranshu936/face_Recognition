import cv2
import numpy as np
from PIL import Image
import os
import json


def load_training_data(directory):
    """
    Load face images and their corresponding labels from the given directory.

    Args:
        directory (str): Path to the directory containing training images.

    Returns:
        list, list: Lists of cropped face images and corresponding labels.
    """
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    face_samples = []
    ids = []

    # Loop through each image in the directory
    for image_path in image_paths:
        img = np.array(Image.open(image_path).convert('L'), 'uint8')
        label = int(os.path.split(image_path)[-1].split("-")[1])  # Extract label from filename

        # Detect faces in the image
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y + h, x:x + w])
            ids.append(label)

    return face_samples, ids


def detect_faces_and_identify(image_path, recognizer):
    """
    Detect faces in an image and identify them using a trained recognizer.

    Args:
        image_path (str): Path to the image to process.
        recognizer (cv2.face.LBPHFaceRecognizer): Trained face recognizer.

    Returns:
        list: List of tuples containing detected name, confidence, and face coordinates.
    """
    # Load the mapping of labels to names
    with open('names.json', 'r') as file:
        names = json.load(file)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Image could not be loaded. Check the file path.")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)
        name = names.get(str(label), "Unknown")

        # Add detection result to the list
        results.append((name, confidence, (x, y, w, h)))

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{name} ({round(confidence)}%)", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the image with detected faces
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


if __name__ == "__main__":
    training_dir = './images/'  # Directory containing training images

    # Initialize the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("\n[INFO] Starting training process...")

    # Load training data
    faces, labels = load_training_data(training_dir)

    # Train the recognizer
    recognizer.train(faces, np.array(labels))

    # Save the trained model
    recognizer.save('trainer.yml')
    print(f"\nTraining completed. {len(np.unique(labels))} unique faces trained.")

    # Load the trained recognizer
    recognizer.read('trainer.yml')

    # Process a test image
    test_image_path = input("Enter the path to the image for identification: ")
    detections = detect_faces_and_identify(test_image_path, recognizer)

    if detections:
        for name, confidence, (x, y, w, h) in detections:
            print(f"Detected: {name} with {confidence:.2f}% confidence at location ({x}, {y}, {w}, {h}).")
    else:
        print(" No faces detected in the image.")
