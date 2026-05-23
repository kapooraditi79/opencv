import cv2;
import numpy as np
from ultralytics import YOLO
import torch

model= YOLO('../yolov10s.pt')
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def detect_people(frame):
    result= model(frame, classes=0)
    detections= result[0].boxes

    if detections is None:
        print("No detection found")
        return None,None
    
    person_boxes= {}
    for i, person in enumerate(detections):
        x1,y1,x2,y2= person.xyxy[0].cpu().numpy()
        person_boxes[i]= [x1,y1,x2,y2]
    
    print("Number of bounding boxes:", len(person_boxes))
    return person_boxes, result

if __name__=='__main__':
    boxes, result= detect_people('../testImage/test6.png')
    annotated_image= result[0].plot()

    cv2.imshow('People Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

