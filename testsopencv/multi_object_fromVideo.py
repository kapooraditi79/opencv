import cv2
from ultralytics import YOLO

cap= cv2.VideoCapture("people.mp4")
model= YOLO("yolov8n.pt")

while True:
    ret, frame= cap.read()      # ret->means  return
    result= model(frame, classes= 0)  # in coco, person is labelled 0
    annotated_frame= result[0].plot()

    cv2.imshow("Annotated Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()