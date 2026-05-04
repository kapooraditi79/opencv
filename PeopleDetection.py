import cv2
import numpy as np
from ultralytics import YOLO

model= YOLO("yolov10s.pt")
cap= cv2.imread('testImage/test6.png')

result= model(cap, classes=0)

annotated_image= result[0].plot()
cv2.imshow('People Detection', annotated_image)


# extract boxes and IDs
detections= result[0].boxes
if detections is None:
    print("No detections found")
person_boxes= {}

if detections is not None:
    for i, detection in enumerate(detections):
        x1, y1, x2, y2= detection.xyxy[0].cpu().numpy()

        # get the tracking id
        # track_id= int(detection.id.item())

        # get confidence score 
        conf= detection.conf.item()

        person_boxes[i]= [x1, y1, x2, y2]

print("This is the number of bounding boxes: ", len(person_boxes))


cv2.waitKey(0)
cv2.destroyAllWindows()


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

