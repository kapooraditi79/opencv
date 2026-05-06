# run_pipeline.py (NEW FILE - single entry point)
import sys
sys.path.append(r'D:\Antares')

import numpy as np
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Import model-loading modules ONCE
from depthpro import get_depth_image, get_feet_depth
from PeopleDetection import detect_people
from pinhole import pixel_to_3d

def get_3d_coordinates(person_boxes, feet_depths, focallength, depth):
    h, w = depth.shape
    fx = fy = focallength
    cx, cy = w/2, h/2
    position_3d = {}
    for person_id, feet_depth in feet_depths.items():
        x1, y1, x2, y2 = person_boxes[person_id]
        feet_x = int((x1 + x2) / 2)
        feet_y = int(y2)
        coords = pixel_to_3d(feet_x, feet_y, feet_depth, fx, fy, cx, cy)
        position_3d[person_id] = coords
    return position_3d

def get_distance(position_3d, eps=1.5, min_samples=4):
    ids = list(position_3d.keys())
    coords = np.array([position_3d[i] for i in ids])
    dist_matrix = euclidean_distances(coords, coords)
    
    # Print distances
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            dist = dist_matrix[i][j]
            print(f"Person {ids[i]} <-> Person {ids[j]}: {dist:.2f}m")
    
    # Cluster
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clusters.fit_predict(dist_matrix)
    labels_dict = {ids[i]: labels[i] for i in range(len(ids))}
    
    print(f"\nLabels (eps={eps}, min_samples={min_samples}): {labels}")
    return labels_dict

def visualize(labels, person_boxes, image_path, output_path='output_labeled.jpg'):
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), 
              (255,0,255), (0,255,255)]
    image_cv = cv2.imread(image_path)
    
    for person_id, label in labels.items():
        x1, y1, x2, y2 = map(int, person_boxes[person_id])
        if label == -1:
            color = (128, 128, 128)
        else:
            color = colors[label % len(colors)]
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_cv, f"G{label}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Grouped People', image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, image_cv)
    print(f"Output saved to {output_path}")

# ===== MAIN PIPELINE (runs everything ONCE) =====
if __name__ == '__main__':
    import time
    start = time.time()
    
    image_path = 'testImage/test6.png'
    
    # Step 1: Depth
    t1 = time.time()
    depth, focallength = get_depth_image(image_path)
    print(f"Depth estimation: {time.time()-t1:.1f}s")
    
    # Step 2: Detection
    t2 = time.time()
    person_boxes = detect_people(image_path)
    print(f"Person detection: {time.time()-t2:.1f}s")
    
    # Step 3: Feet depth
    t3 = time.time()
    feet_depths = get_feet_depth(depth, person_boxes)
    print(f"Feet depth sampling: {time.time()-t3:.1f}s")
    
    # Step 4: 3D coordinates
    t4 = time.time()
    position_3d = get_3d_coordinates(person_boxes, feet_depths, focallength, depth)
    print(f"3D conversion: {time.time()-t4:.1f}s")
    
    # Step 5: Clustering (try multiple eps values)
    for eps in [0.8, 1.2, 1.5, 2.0]:
        labels = get_distance(position_3d, eps=eps, min_samples=4)
    
    # Step 6: Visualize with best eps
    labels_final = get_distance(position_3d, eps=1.5, min_samples=4)
    visualize(labels_final, person_boxes, image_path)
    
    print(f"\nTotal pipeline time: {time.time()-start:.1f}s")