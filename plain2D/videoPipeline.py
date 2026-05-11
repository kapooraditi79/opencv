import cv2
from ultralytics import YOLO
import sys
import os
sys.path.append(r'D:\Antares\plain2D')
from VidyoloDetect import detect_people
from penalizedDist import extract_metrics, get_penalized_dist
from sklearn.cluster import DBSCAN
import numpy as np



video_path= r'..\testVideo\test1.mp4'
filename = os.path.basename(video_path)
os.makedirs("VidOutput", exist_ok=True)

OUTPUT_PATH = os.path.join("VidOutput", filename)
print("Saving to:", OUTPUT_PATH)


k_val= 0.5
eps=150
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
    complete_penalized_map, _=get_penalized_dist(px_dist, h_ratios, total_boxes)

    dist_matrix= complete_penalized_map[k_val]

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


# import cv2
# import os
# import json
# import numpy as np

# from sklearn.cluster import DBSCAN

# from VidyoloDetect import detect_people
# from penalizedDist import extract_metrics, get_penalized_dist

# from groupTracking import (
#     build_clusters,
#     match_groups
# )

# # ============================================================
# # VIDEO SETTINGS
# # ============================================================

# video_path = r'..\testVideo\test1.mp4'

# filename = os.path.basename(video_path)

# OUTPUT_PATH = os.path.join(
#     "VidOutput",
#     f"tracked_{filename}"
# )

# HISTORY_PATH = os.path.join(
#     "VidOutput",
#     "tracking_history.json"
# )

# # ============================================================
# # DBSCAN PARAMETERS
# # ============================================================

# k_val = 0.5
# eps = 150
# min_sample = 3

# # ============================================================
# # GROUP TRACKING PARAMETERS
# # ============================================================

# MATCH_THRESHOLD = 0.5

# MAX_INACTIVE_FRAMES = 30

# # ============================================================
# # COLORS
# # ============================================================

# GROUP_COLORS = [
#     (255, 0, 0),
#     (0, 255, 0),
#     (0, 0, 255),
#     (255, 255, 0),
#     (255, 0, 255),
#     (0, 255, 255),
#     (128, 0, 128),
#     (255, 128, 0),
#     (0, 128, 128),
#     (128, 128, 0),
# ]

# NOISE_COLOR = (180, 180, 180)

# # ============================================================
# # VIDEO SETUP
# # ============================================================

# cap = cv2.VideoCapture(video_path)

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fps = int(cap.get(cv2.CAP_PROP_FPS))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# out = cv2.VideoWriter(
#     OUTPUT_PATH,
#     fourcc,
#     fps,
#     (width, height)
# )

# # ============================================================
# # GROUP MEMORY
# # ============================================================

# active_groups = {}

# next_group_id = 0

# frame_idx = 0

# frame_count = 0

# # ============================================================
# # TRACKING HISTORY STORAGE
# # ============================================================

# tracking_history = {}

# # ============================================================
# # MAIN LOOP
# # ============================================================

# while cap.isOpened():

#     success, frame = cap.read()

#     if not success:
#         break

#     frame_idx += 1

#     frame_count += 1

#     # ========================================================
#     # PROCESS EVERY 5TH FRAME
#     # ========================================================

#     if frame_count % 5 != 0:

#         out.write(frame)

#         cv2.imshow("Video", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         continue

#     # ========================================================
#     # YOLO + TRACKING
#     # ========================================================

#     person_boxes, result = detect_people(frame)

#     # ========================================================
#     # HANDLE NO DETECTIONS
#     # ========================================================

#     if person_boxes is None or len(person_boxes) < 3:

#         out.write(frame)

#         cv2.imshow("Video", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         continue

#     # ========================================================
#     # DISTANCE METRICS
#     # ========================================================

#     px_dist, h_ratios, total_boxes, person_ids = extract_metrics(
#         person_boxes
#     )

#     complete_penalized_map, _ = get_penalized_dist(
#         px_dist,
#         h_ratios,
#         total_boxes
#     )

#     dist_matrix = complete_penalized_map[k_val]

#     # ========================================================
#     # DBSCAN
#     # ========================================================

#     clustering = DBSCAN(
#         eps=eps,
#         min_samples=min_sample,
#         metric='precomputed'
#     )

#     labels = clustering.fit_predict(dist_matrix)

#     # ========================================================
#     # BUILD CURRENT CLUSTERS
#     # ========================================================

#     current_clusters = build_clusters(
#         person_ids,
#         labels
#     )

#     # ========================================================
#     # MATCH TO OLD GROUPS
#     # ========================================================

#     assigned_groups, next_group_id = match_groups(
#         current_clusters,
#         active_groups,
#         frame_idx,
#         next_group_id,
#         threshold=MATCH_THRESHOLD
#     )

#     # ========================================================
#     # REMOVE EXPIRED GROUPS
#     # ========================================================

#     expired = []

#     for group_id, group_data in active_groups.items():

#         if frame_idx - group_data['last_seen'] > MAX_INACTIVE_FRAMES:

#             expired.append(group_id)

#     for group_id in expired:

#         del active_groups[group_id]

#     # ========================================================
#     # STORE TRACKING HISTORY
#     # ========================================================

#     tracking_history[frame_idx] = {}

#     for group_id, members in assigned_groups.items():

#         tracking_history[frame_idx][str(group_id)] = {

#             "members": list(map(int, members)),

#             "boxes": {

#                 str(person_id): list(
#                     map(float, person_boxes[person_id])
#                 )

#                 for person_id in members

#                 if person_id in person_boxes
#             }
#         }

#     # ========================================================
#     # BUILD PERSON -> GROUP MAP
#     # ========================================================

#     person_to_group = {}

#     for group_id, members in assigned_groups.items():

#         for person_id in members:

#             person_to_group[person_id] = group_id

#     # ========================================================
#     # DRAW BOXES
#     # ========================================================

#     for idx, person in person_boxes.items():

#         x1, y1, x2, y2 = map(int, person)

#         # ====================================================
#         # GET PERSISTENT GROUP ID
#         # ====================================================

#         if idx in person_to_group:

#             group_id = person_to_group[idx]

#         else:

#             group_id = -1

#         # ====================================================
#         # COLORS
#         # ====================================================

#         if group_id == -1:

#             color = NOISE_COLOR

#             group_name = "Noise"

#         else:

#             color = GROUP_COLORS[
#                 group_id % len(GROUP_COLORS)
#             ]

#             group_name = f"G{group_id}"

#         # ====================================================
#         # DRAW RECTANGLE
#         # ====================================================

#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             color,
#             2
#         )

#         # ====================================================
#         # DRAW TEXT
#         # ====================================================

#         cv2.putText(
#             frame,
#             f"P{idx}:{group_name}",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             color,
#             2
#         )

#     # ========================================================
#     # WRITE FRAME
#     # ========================================================

#     out.write(frame)

#     cv2.imshow("Video", frame)

#     # ========================================================
#     # EXIT
#     # ========================================================

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ============================================================
# # CLEANUP
# # ============================================================

# cap.release()

# out.release()

# cv2.destroyAllWindows()

# # ============================================================
# # SAVE TRACKING HISTORY
# # ============================================================

# with open(HISTORY_PATH, "w") as f:

#     json.dump(
#         tracking_history,
#         f,
#         indent=4
#     )

# print(f"\nSaved output video to:\n{OUTPUT_PATH}")

# print(f"\nSaved tracking history to:\n{HISTORY_PATH}")