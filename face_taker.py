import numpy as np
import json
import cv2
import os

def create_directory(directory: str):
    """
    This function ensures the specified directory exists. If it doesn't exist, it creates it.
    
    Parameters:
        directory (str): The path of the directory to be created.
    """
    try:
        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        # In case of an error (e.g., permission issues), print an error message
        print(f"Error creating directory {directory}: {e}")

def get_face_id(directory: str) -> int:
    """
    This function retrieves the next available user ID based on the existing image files in the directory.
    
    Parameters:
        directory (str): The directory containing the user images.
        
    Returns:
        int: The next available face ID.
    """
    # Collect user IDs from existing image filenames
    user_ids = [int(os.path.split(f)[-1].split("-")[1]) for f in os.listdir(directory)]
    
    # Sort and remove duplicates to get a list of unique IDs
    user_ids = sorted(set(user_ids))
    
    # Determine the next available face ID
    max_user_ids = 1 if not user_ids else max(user_ids) + 1
    
    # Find the first available ID by checking for missing IDs in the sequence
    for i in range(max_user_ids):
        if i not in user_ids:
            return i
    return max_user_ids

def save_name(face_id: int, face_name: str, filename: str):
    """
    This function saves the face ID and name mapping to a JSON file.
    
    Parameters:
        face_id (int): The ID assigned to the user.
        face_name (str): The name of the user.
        filename (str): The JSON file where the mapping will be saved.
    """
    names_json = {}
    
    # Load the existing names mapping if the file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            names_json = json.load(f)
    
    # Add the new face ID and name mapping to the dictionary
    names_json[face_id] = face_name
    
    # Save the updated names mapping to the JSON file
    with open(filename, 'w') as f:
        json.dump(names_json, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # Set paths and filenames
    directory = 'images'
    cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
    names_json_filename = 'names.json'

    # Ensure the 'images' directory exists
    create_directory(directory)
    
    # Load the pre-trained Haar Cascade Classifier for face detection
    faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)
    
    # Open the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    
    # Set camera resolution to 640x480
    cam.set(3, 640)
    cam.set(4, 480)

    # Prompt the user to enter their name
    face_name = input('Enter user name: ')
    
    # Get the next available face ID
    face_id = get_face_id(directory)
    
    # Save the face ID and name mapping in a JSON file
    save_name(face_id, face_name, names_json_filename)
    
    # Inform the user to look at the camera
    print('Initializing face capture. Look at the camera and wait...')
    
    # Variable to count the number of captured images
    count = 0
    
    # Start capturing images
    while True:
        # Capture a frame from the camera
        ret, img = cam.read()
        
        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Loop through all detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Increment the counter for captured face images
            count += 1
            
            # Save the face image as a grayscale image in the 'images' directory
            cv2.imwrite(f'./images/Users-{face_id}-{count}.jpg', gray[y:y + h, x:x + w])
            
            # Display the image with face rectangles
            cv2.imshow('image', img)

        # If the Escape key is pressed, stop the capture
        if cv2.waitKey(100) & 0xFF < 30 or count >= 30:
            break

    # Notify the user that the face capture is complete
    print('Success! Exiting Program.')
    
    # Release the camera resource
    cam.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
