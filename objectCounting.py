# tracking is very difficult for the fast moving objects
# tracking is done using IOU= area_of_intersection/ area_of_union

import cv2
from ultralytics import YOLO
import numpy as np

model= YOLO('yolov8n.pt')

cap= cv2.VideoCapture(0)

# a variable to store the unique ids
unique_ids= set()

while True:
    ret, frame= cap.read()
    results= model.track(frame, classes=[0,39], persist=True, verbose= False) # just formatting in terminal) # if you were tracking bottles, you 'd have used classes= [39]
    # results from this
    annotated_frame= results[0].plot()

    # if there are boxes and ids in the annotated frame
    if results[0].boxes and results[0].boxes.id is not None:
        ids= results[0].boxes.id.numpy()
        for oid in ids:
            unique_ids.add(oid)
        # text for object id
        cv2.putText(annotated_frame, f'Count: {len(unique_ids)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.imshow('Object Tracking', annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




