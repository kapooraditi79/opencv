import cv2
from ultralytics import YOLO
import torch

model = YOLO('../yolov10s.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def detect_bicycle(frame):
    result= model.predict(
        frame,
        classes= 1,
        conf= 0.5,
        iou= 0.5
    )

    detection = result[0].boxes
    if detection is None:
        print('No detection found')
        return None, None

    bicycle_boxes= {}
    for bicycle in detection:

        if bicycle.id is None:
            continue

        track_id= int(bicycle.id.item())
        x1, x2,y1,y2= bicycle.xyxy[0].cpu().numpy()
        bicycle_boxes[track_id]= [x1,y1,x2,y2]

        print('Number of bounding boxes found: ', len(bicycle_boxes))

    return bicycle_boxes, result


if __name__=='__main__':
    video_path= r'..\testVideo\bike.mp4'
    cap= cv2.VideoCapture(video_path)
    frame_count=0
    while cap.isOpened():
        ret, frame= cap.read()
        if not ret:
            break

        frame_count+=1

        if frame_count%4!=0:
            continue
        boxes, results= detect_bicycle(frame)
        print('Number of bicycle boxes: ', len(boxes))

        annotated_frame= results[0].plot()

        cv2.imshow('Bicycle Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
