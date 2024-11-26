import cv2
import numpy as np
from PIL import Image
import os

# Function to load images and labels for training
def get_images_and_labels(path):
    # List all image file paths in the specified directory
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []  # Initialize lists to hold face samples and corresponding labels

    # Iterate over all image paths
    for image_path in image_paths:
        # Open the image, convert it to grayscale, and then convert it to a numpy array
        img = np.array(Image.open(image_path).convert('L'), 'uint8')
        
        # Extract the ID (user identifier) from the image file name
        id = int(os.path.split(image_path)[-1].split("-")[1])
        
        # Use the Haar cascade classifier to detect faces in the grayscale image
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(img)
        
        # Iterate over the detected faces and extract face regions
        for (x, y, w, h) in faces:
            # Append the face region (cropped from the original image) and the corresponding label (ID)
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)
    
    # Return the list of face samples and corresponding labels
    return face_samples, ids

# Main program execution
if __name__ == "__main__":
    path = './images/'  # Path where the images are stored
    
    # Create an LBPH face recognizer object
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Print message indicating the training process has started
    print("\n[INFO] Training...")
    
    # Get the face samples and corresponding labels from the images in the specified directory
    faces, ids = get_images_and_labels(path)
    
    # Train the recognizer using the face samples and labels
    recognizer.train(faces, np.array(ids))
    
    # Save the trained model to a file (trainer.yml)
    recognizer.write('trainer.yml')
    
    # Print the number of faces trained and exit the program
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
