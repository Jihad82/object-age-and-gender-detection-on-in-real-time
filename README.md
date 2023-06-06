# Object, Age and Gender Detection Using OpenCV's Deep Neural Networks

This is a Python project that performs real-time age and gender detection on faces using OpenCV's deep neural networks. The script also performs object detection on the full-sized frame to detect objects within the face region.

The script uses pre-trained models, which are loaded into memory using the `cv2.dnn.readNet` method. These models include:

- `age_net.caffemodel` and `age_deploy.prototxt` for age detection
- caffemodel: age_net.caffemodel |caffemodel_url: https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
- `gender_net.caffemodel` and `gender_deploy.prototxt` for gender detection
- caffemodel: gender_net.caffemodel | caffemodel_url: https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
- `opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt` for face detection
- `MobileNetSSD_deploy.caffemodel` and `MobileNetSSD_deploy.prototxt` for object detection

The script captures frames from the default camera (index 0) using `cv2.VideoCapture(0)` and performs the following steps for each frame:

1. Resizes the frame to a smaller size for better optimization.
2. Performs face detection on the smaller frame using the face detection model.
3. If a face is detected, performs gender and age detection on the face region using their respective models.
4. Performs object detection on the full-sized frame using the object detection model within the face region.
5. Draws the results of the detections onto the original frame and displays them in two windows: one for age and gender detection and the other for object detection.
6. Prints the time taken to process each frame.

## Requirements
To run this project, you need to have the following installed:

- Python 3.x
- OpenCV
- NumPy

You can install the required packages by running the following command:

```
pip install opencv-python numpy
```

## Usage
To run the script, execute the following command in your terminal:

```
python age_gender_detection.py
```

This will launch the script and start capturing frames from your default camera.

## Credits
This project was developed by Dev Jihad
