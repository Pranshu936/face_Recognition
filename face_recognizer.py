import cv2
import numpy as np
import json

def load_trained_model(model_path='trainer.yml'):
    """
    Load the pre-trained face recognition model.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer

def load_names(names_file='names.json'):
    """
    Load user names from a JSON file.
    """
    with open(names_file, 'r') as file:
        names_data = json.load(file)
    return list(names_data.values())

def initialize_camera():
    """
    Initialize and return the camera object with preset configurations.
    """
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set frame width to 640 pixels
    cam.set(4, 480)  # Set frame height to 480 pixels
    return cam

def detect_faces(image, face_cascade, min_size=(0.1, 0.1)):
    """
    Detect faces in a given image using the specified cascade classifier.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_w, min_h = int(min_size[0] * image.shape[1]), int(min_size[1] * image.shape[0])
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(min_w, min_h))
    return faces, gray

def recognize_face(face_recognizer, gray_face):
    """
    Recognize a face using the pre-trained recognizer.
    """
    label, confidence = face_recognizer.predict(gray_face)
    return label, confidence

def main():
    # Load the trained recognizer and user names
    recognizer = load_trained_model('trainer.yml')
    names = load_names('names.json')
    
    # Initialize face detection and camera
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = initialize_camera()
    
    # Set the minimum face size relative to the frame size
    min_size = (0.1, 0.1)  # 10% of frame width and height
    
    print("[INFO] Starting face recognition...")

    while True:
        # Capture a frame from the camera
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        
        # Detect faces in the captured frame
        faces, gray = detect_faces(img, face_cascade, min_size)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the region of interest (ROI) for face recognition
            gray_face = gray[y:y + h, x:x + w]
            
            # Recognize the face
            label, confidence = recognize_face(recognizer, gray_face)
            
            # Check if recognition confidence is above threshold (51%)
            if confidence > 51:
                try:
                    name = names[label]
                    confidence_text = f"{round(confidence)}%"
                except IndexError:
                    name = "Unknown"
                    confidence_text = "N/A"
            else:
                name = "Unknown"
                confidence_text = "N/A"
            
            # Display recognized name and confidence on the image
            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        # Show the frame with rectangles and recognized faces
        cv2.imshow('Face Recognition', img)
        
        # Exit the loop if the Escape key (27) is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    # Clean up and release resources
    print("[INFO] Exiting program...")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
