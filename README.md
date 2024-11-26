# Real-Time Face Recognition with OpenCV

This project demonstrates how to create a fast and efficient real-time face recognition system using Python and OpenCV. The application can detect faces, capture face images, train a model, and recognize faces in real-time.



## Features
- Capture face images and associate them with user names.
- Train a face recognition model using captured images.
- Recognize faces in real-time with user-specific labels.
- Recognize faces from image  



## Getting Started

This project requires a webcam to capture images and perform face recognition. Ensure your camera is set up and working before running the scripts.



## Scripts Overview

### 1. `face_taker.py`
This script captures **30 images** of a user's face and assigns a unique ID to the user. It saves:
- **Images** in the `images` folder.
- A mapping of user IDs and names in the `names.json` file.



### 2. `face_train.py`
This script trains a face recognition model using the images captured in the `images` folder. It:
- Uses the **LBPH (Local Binary Patterns Histogram)** algorithm.
- Saves the trained model to the `trainer.yml` file.



### 3. `face_recognizer.py`
This is the main face recognition script. It:
- Loads the trained model (`trainer.yml`) and `names.json`.
- Recognizes faces in real-time and displays user names with confidence levels on the video feed.



## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python libraries:
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python
   pip install pillow
   ```



## How to Use

1. **Capture Face Images**:
   Run the `face_taker.py` script:
   ```bash
   python face_taker.py
   ```
   - Enter your name when prompted.
   - Look at the camera until 30 images are captured.

2. **Train the Model**:
   Run the `face_train.py` script:
   ```bash
   python face_train.py
   ```
   - This will generate the `trainer.yml` file.

3. **Recognize Faces**:
   Run the `face_recognizer.py` script:
   ```bash
   python face_recognizer.py
   ```
   - The script will recognize faces and display user names and confidence levels.



## Folder Structure
```
.
├── images/                 # Stores captured face images
├── trainer.yml             # Trained face recognition model
├── names.json              # Stores user ID-name mappings
├── face_taker.py           # Script for capturing face images
├── face_train.py           # Script for training the model
├── face_recognizer.py      # Script for real-time face recognition
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
└── README.md               # Project documentation
```



## Requirements

- **Python 3.6+**
- OpenCV for face detection and recognition:
  ```bash
  pip install opencv-python
  pip install opencv-contrib-python
  ```
- **Pillow** for image handling:
  ```bash
  pip install pillow
  ```



## Notes

- Ensure your face is centered in the frame during image capture.
- The project uses the **LBPH (Local Binary Patterns Histogram)** algorithm for robust recognition.




