import array as arr
import cv2
import pandas as pd
import numpy as np
from YOLOv7 import YOLOv7

#global variables
src = None
focal = 4
baseline = 120
pixtomm = 0.2645833333

def Merge_toData(boxes, class_ids):
    data = arr.array
    data = pd.DataFrame(boxes)
    data[4] = class_ids
    return data

#Xác định tâm của bbox
def Define_centerpoint(data, class_ids):
    x_center = 0
    y_center = 0
    center_point = []

    data = data.reset_index()
    for index, row in data.iterrows():
            x_center = (data[2][index] - data[0][index]) / 2
            y_center = (data[3][index] - data[1][index]) / 2
            center_point.append([x_center, class_ids[index]])
    return center_point


def column(matrix, i):
    return [row[i] for row in matrix]
##Xác định khoảng cách của vật
def detect_Distance(centerPoint_Object1, centerPoint_Object2):
    #Tạo mảng 2 chiều dạng [xl, xr]
    get1 = np.asarray(centerPoint_Object1)
    print ("get1",get1)
    get2 = np.asarray(centerPoint_Object2)
    print ("get2",get2)
    get_xl= get1[:,0]
    get_xr = get2[:,0]
    a = len(get_xl)
    b = len(get_xr)
    if a >= b:
        distance =  []
    else:
        distance = []
    print ("get_xl",get_xl)
    print ("get_xr",get_xr)
    try:
        for i,item in enumerate(get_xl,0):
            distance = np.append(distance, ((baseline*focal* pixtomm)/ (abs(get_xl[i] - get_xr[i])))) 
    except IndexError:
        print('distance is empty')
    return distance

def main():
    
    model_path = "models/yolov7-tiny_Nx3x736x1280.onnx"
    yolov7_detector_left = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
    yolov7_detector_right = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
    video_capture_0 = cv2.VideoCapture(0)
    video_capture_1 = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret0, frame0 = video_capture_0.read()
        ret1, frame1 = video_capture_1.read()
        
        boxes_left, scores_left, class_ids_left = yolov7_detector_left(frame0)
        boxes_right, scores_right, class_ids_right = yolov7_detector_right(frame1)

        combined_img0 = yolov7_detector_left.draw_detections(frame0)
        combined_img1 = yolov7_detector_right.draw_detections(frame1)
       
        x = len(boxes_left)
        y = len(boxes_right)
        if x >0  and y >0  :
            data_left = Merge_toData(boxes_left, class_ids_left)
            print(data_left)
            data_right = Merge_toData(boxes_right, class_ids_right)
            print(data_right)
            
            center_left = Define_centerpoint(data_left, class_ids_left)
            print(center_left)
            center_right = Define_centerpoint(data_right, class_ids_right)
            print(center_right)
            distance = detect_Distance(center_left, center_right)
            print("khoang cach:", distance, "m")
            temp= np.asarray(data_left)
            x_temp = temp[:,2]
            y_temp = temp[:,1]
            try:
                for i, items in enumerate(x_temp,0):
                    x = int(x_temp[i]-150)
                    y = int(y_temp[i])
                    dis = str(int(distance[i]))
                    print("distance", distance[i])
                    cv2.putText(combined_img0,dis, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    print("x,y", x, y, dis)
                    cv2.putText(combined_img0,("mm"), (x+100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            except IndexError:
                print("obj is not found")
        print('distance is empty')        
        if (ret1):
                # Display the resulting frame
            cv2.imshow('Cam 1', combined_img1)
        if (ret0):
                # Display the resulting frame
            cv2.imshow('Cam 0', combined_img0)    
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

if __name__ == "__main__":
    main()







