from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8m.pt", "v8")

def runyolo(img):
    # predict on an image
    detection_output = model.predict(source=img, conf=0.25,show=False)
    return detection_output[0]



#img = cv2.imread("bus.jpg")  # For Video


#
# def yolorun(img):
#     model = YOLO("../Yolo-Weights/yolov8m.pt")
#     results = model(img, stream=False)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#     return results[0]

    #cv2.imshow("Image", img)
    #cv2.waitKey(0)






