import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict

model= YOLO('yolov8n.pt')
cap= cv2.VideoCapture('peeps.mp4')

# tracking center of geometry
id_map={}
nex_id = 1

trail= defaultdict(lambda: deque(maxlen=30))
# weather the person has appeared or not dictionary
appear= defaultdict(int)

while True:
    ret, frame= cap.read()
    results= model.track(frame, classes=[0], persist= True, verbose= False)
    annotated_frame= frame.copy()   #[image+centerofmass+rectangle+text]

    if results[0].boxes.id is not None:
        boxes= results[0].boxes.xyxy.numpy()   # get the boxes coordinates
        ids= results[0].boxes.id.numpy()      # get the ids in the boxes

        for box, oid in zip(boxes, ids):
            x1,y1,x2,y2= map(int, box)
            cx,cy= (x1+x2)//2, (y1+y2)//2    # centroids

            # among the boxes that are appearing
            appear[oid] += 1

            # now see if this person has been added in the id_map
            if appear[oid]>=5 and oid not in id_map:
                id_map[oid]= nex_id
                nex_id +=1
            
            # now check if the person is in the id map
            if oid in id_map:
                sid= id_map[oid]
                trail[oid].append((cx, cy))
                # we are going to show a trail for 30 frames only, even if the object is appearing in a 100 frames
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(annotated_frame, f'ID: {sid}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
                cv2.circle(annotated_frame, (cx,cy), 4, (255,0,0), -1)

        cv2.imshow('Tracking', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

                
