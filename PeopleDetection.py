import cv2
import numpy as np
from ultralytics import YOLO

model= YOLO("yolov10s.pt")
cap= cv2.imread('testImage/test6.png')

result= model(cap, classes=0)

annotated_image= result[0].plot()
cv2.imshow('People Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # for the video 
# cap= cv2.VideoCapture('testVideo/test2.mp4')

# while True:
#     ret, frame= cap.read()
#     result= model(frame, classes=0)
#     annotated_frame= result[0].plot()

#     cv2.imshow('People Detection', annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# people detected successfully!!

# now we will use the depth pro model

