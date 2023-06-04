from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np

cap = cv2.VideoCapture("Videos/simulator.mp4")  # For Video
model = YOLO("Yolo-Weights/yolov8m.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    overlay = img.copy()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or  currentClass == "person" and conf > 0.3:
                cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), (255, 0, 255), -1)  # A filled rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 25), 3)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(20, 20, 25))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                                          offset=3, colorR=(100, 0, 255))

                alpha = 0.1  # Transparency factor.
                # Following line overlays transparent rectangle over the image
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # if currentClass == "person" and conf > 0.3:
            #     cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), (255, 0, 255), -1)  # A filled rectangle
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 25), 3)
            #     cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
            #                        offset=3, colorR=(100, 0, 255))
            #     alpha = 0.03  # Transparency factor.
            #     # Following line overlays transparent rectangle over the image
            #     img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    fps = 1 / (new_frame_time - prev_frame_time)
    fps = math.ceil(fps)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f' {fps}', (max(0, 20), max(35, 30)), scale=1, thickness=1, colorR=(125, 70, 89))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
