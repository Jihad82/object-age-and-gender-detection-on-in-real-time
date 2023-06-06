import cv2
import math
import time
import numpy as np

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)

    return frameOpencvDnn, bboxes


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

objectProto = "MobileNetSSD_deploy.prototxt"
objectModel = "MobileNetSSD_deploy.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
objectList = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor', 'phone', 'paper', 'pen']


ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
objectNet = cv2.dnn.readNet(objectModel, objectProto)

cap = cv2.VideoCapture(0)
padding = 20

while cv2.waitKey(1) < 0:
    
    t = time.time()
    hasFrame, frame = cap.read()

    if not hasFrame:
        cv2.waitKey()
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    frameFace, bboxes = getFaceBox(faceNet, small_frame)
    if not bboxes:
        print("No face detected, checking next frame")
        continue

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    objectNet.setInput(blob)
    detections = objectNet.forward()

    for bbox in bboxes:
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender: {}, confidence = {:.3f}".format(
            gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age: {}, confidence = {:.3f}".format(age, agePreds[0].max()))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                class_name = objectList[class_id]
                box = detections[0, 0, i, 3:7] * \
                    np.array([frameWidth, frameHeight,
                             frameWidth, frameHeight])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                cv2.rectangle(frame, (start_x, start_y),
                              (end_x, end_y), (0, 255, 0), 2)
                label = "{}: {:.2f}%".format(class_name, confidence * 100)
                y = start_y - 15 if start_y - 15 > 15 else start_y + 15
                cv2.putText(frame, label, (start_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        label = "{}, {}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Age and Gender Detection", frameFace)
        cv2.imshow("Object Detection", frame)

    print("Time: {:.3f}".format(time.time() - t))
