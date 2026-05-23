# file 4
# phase a- naive approch
# contains
    # flickering clusters
    # label swapping
    # group fragmentation
    # intermittent detections
    # unstable noise assignment
import cv2
# from ultralytics import YOLO
import sys
import os
sys.path.append(r'D:\Antares\plain2D')
from VidyoloDetect import detect_people
from penalizedDist import extract_metrics, get_penalized_dist
from sklearn.cluster import DBSCAN
# import numpy as np

video_path= r'..\testVideo\test1.mp4'
filename = os.path.basename(video_path)
os.makedirs("VidOutput", exist_ok=True)

OUTPUT_PATH = os.path.join("VidOutput", filename)
print("Saving to:", OUTPUT_PATH)


k_vals= [0.5]
# eps=100
eps=120
min_sample=3

GROUP_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

NOISE_COLOR = (180, 180, 180)

# now running the pipeline per frame
# model= YOLO('../yolov10s.pt')

cap= cv2.VideoCapture(video_path)

# extract the video metrics
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_count=0

while cap.isOpened():
    success, frame= cap.read()
    if not success:
        break

    # for speed
    frame_count+=1
    if frame_count%5!=0:
        out.write(frame)
        continue

    person_boxes, result= detect_people(frame)

    if person_boxes is None or len(person_boxes)<3:
        # straight up draw the noise boxes
        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        continue

    px_dist, h_ratios, total_boxes= extract_metrics(person_boxes)
    # k_vals=[0.5]
    complete_penalized_map, _=get_penalized_dist(px_dist, h_ratios, total_boxes, k_vals=k_vals)
    dist_matrix= complete_penalized_map[k_vals[0]]

    # now do DBSCAN

    clustering= DBSCAN(
        eps= eps,
        min_samples=min_sample,
        metric='precomputed'
    )

    labels= clustering.fit_predict(dist_matrix)

    # draw boxes

    for idx, person in person_boxes.items():
        x1, y1,x2,y2= map(int, person)  
        label= labels[idx]
            # convert float to int
        if label == -1:
            color = NOISE_COLOR
            group_name = "Noise"

        else:
            color = GROUP_COLORS[label % len(GROUP_COLORS)]
            group_name = f"G{label}"

        # rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # text
        cv2.putText(
            frame,
            f"P{idx}:{group_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    out.write(frame)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()

cv2.destroyAllWindows() 
