# file 5
# maintaining the DBSCAN cluster labels as persistant identities in frames
# ex- in file 4- Cluster 0 in frame 10 has no relationship to cluster 0 in frame 11.
# improving this

# maintain stable group IDs across time
import cv2
import numpy as np
from ultralytics import YOLO
import torch

model= YOLO('../yolov10s.pt')
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def detect_people(frame):
    result= model.track(
        frame,
        tracker= 'bytetrack.yaml',
        classes=0,
        conf= 0.45,
        iou= 0.5
    )

    detections= result[0].boxes
    if detections is None:
        print('No detections found')
        return None, None

    person_boxes= {}

    for person in detections:
        if person.id is None:
            continue

        track_id= int(person.id.item())
        x1,y1,x2,y2= person.xyxy[0].cpu().numpy()
        person_boxes[track_id]= [x1,y1,x2,y2]

    print('Number of bounding boxes found: ', len(person_boxes))
    return person_boxes, result

if __name__=='__main__':
    boxes, result= detect_people('../testImage/test6.png')
    annotated_image= result[0].plot()

    cv2.imshow('People Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
