import cv2
import array as arr
import numpy as np
import pandas as pd
from imread_from_url import imread_from_url

from YOLOv7 import YOLOv7

# Initialize YOLOv7 object detector
model_path = "models/yolov7-tiny_Nx3x736x1280.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.3, iou_thres=0.5)

# Read image

img = cv2.imread("D:\project\ONNX-YOLOv7-Object-Detection-main\horses.jpg")
# img_url = "https://staticgthn.kinhtedothi.vn/Uploaded/ducthoatgt/2021_01_21/1_PLEP.jpg"
# img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids = yolov7_detector(img)

print(boxes)
print(class_ids)

#Merge class_ids and boxes into data
data = arr.array
data = pd.DataFrame(boxes)
data[4] = class_ids 
print(data)

#xac dinh khoang cach
x_center = 0
y_center = 0
center_point = []
count = 0
print(data[4][count])

data = data.reset_index()
for index, row in data.iterrows():
        x_center = (data[2][count] - data[0][count]) / 2
        y_center = (data[3][count] - data[1][count]) / 2
        # print(x_center, y_center)
        center_point.append([x_center, y_center])
        count += 1

print(center_point)

# Draw detections
combined_img = yolov7_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
