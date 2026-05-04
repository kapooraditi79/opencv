import cv2
from ultralytics import YOLO
import numpy as np

model= YOLO("yolov8n-seg.pt")

cap= cv2.VideoCapture('peeps.mp4')

while True:
    ret, frame= cap.read()
    results= model.track(frame, persist= True, verbose= False)

    for r in results:
        annotated_frame= frame.copy()

        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks= r.masks.data.cpu().numpy()
            boxes= r.boxes.xyxy.numpy()
            ids= r.boxes.id.numpy()

            for i, mask in enumerate(masks):
                person_id= ids[i]
                x1,y1, x2, y2= boxes[i]
                mask_resized= cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
                contours, _= cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (255,0,0),2)

        cv2.imshow('Segmentation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


                # 
