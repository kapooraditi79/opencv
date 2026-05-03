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

# people detected successfully!!

# now we will use the depth pro model

