import cv2
from ultralytics import YOLO

model= YOLO('yolov10s.pt')
 
cap= cv2.VideoCapture('dogs.mp4')

while True: 
    ret, frame= cap.read()
    result= model.track(frame, classes=[16])
    annotated_frame= result[0].plot()

    cv2.imshow('Dog detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()