from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np

# cap = cv2.VideoCapture(1)  # For Webcam

cap = cv2.VideoCapture("Videos/crazy.mp4")  # For Video
# cap.set(3, 400)
# cap.set(4, 300)

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

            if currentClass == "horse" and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorR=(0, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))


            if currentClass == "stop sign" and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorR=(0, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))

            if currentClass == "bus" and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorR=(0, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))

            if currentClass == "traffic light" and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3, colorR=(100, 0, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 255))

            if currentClass == "person" and conf > 0.3:
                 cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                     thickness=1, offset=3, colorR=(227, 111, 146))
                 cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(227, 111, 146))

                #cv2.rectangle(img, (x1+1, y1+1), (x2-1, y2-1), (210, 81, 146), 2)
                #center_coordinates = ((x2-x1)/2, (y2-y1)/2)
                # height, width, channels = img.shape
                # cx = np.divide((x1+x2), 2)/width
                # cy = np.divide((y1+y2), 2)/height
                # print(cx, cy)


    fps = 1 / (new_frame_time - prev_frame_time)
    fps = math.ceil(fps)
    prev_frame_time = new_frame_time

    cvzone.putTextRect(img, f' {fps}', (max(0, 20), max(35, 30)), scale=1, thickness=1, colorR=(125, 70, 89))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
