import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.append(r'D:\Antares')  

model= YOLO("yolov10s.pt")

# # for the video  
# cap= cv2.VideoCapture('testVideo/test2.mp4')

# while True:
#     ret, frame= cap.read()
#     result= model.track(frame, classes=0)
#     annotated_frame= result[0].plot()

#     cv2.imshow('People Detection', annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# people detected successfully!!

# now we will use the depth pro model



def detect_people(image_path):
    cap= cv2.imread(image_path)
    result= model(cap, classes=0)
    detections= result[0].boxes

    if detections is None:
        print('No detection found')
        return None
    person_boxes= {}

    for i, detection in enumerate (detections):
        x1, y1, x2, y2= detection.xyxy[0].cpu().numpy()
        person_boxes[i]= [x1, y1, x2, y2]
    print("Number of bounding boxes:", len(person_boxes))
    return person_boxes

if __name__== '__main__':
    boxes= detect_people('testImage/test6.png')
    cap= cv2.imread('testImage/test6.png')

    result= model(cap, classes=0)
    annotated_image= result[0].plot()
    cv2.imshow('People Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
