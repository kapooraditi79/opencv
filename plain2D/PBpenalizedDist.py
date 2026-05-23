# file 6
import math
import sys
sys.path.append(r'D:\Antares\plain2D')
from PBVideoDetect import detect_people
from ultralytics import YOLO
import numpy as np

# the ids are not contiguous now, due to bytetracking
# create mappings 
# ex: track ids= [5,17,20]
        # id to idx= {5: 0, 
        #             17:1,
        #             20:2  }

def extract_metrics(person_boxes):
    px_dist= {}
    h_ratios= {}
    track_ids= list(person_boxes.keys())
    id_to_idx = {
        track_id: idx
        for idx, track_id in enumerate(track_ids)
    }
    for track_i, boxes_i in person_boxes.items():
        i= id_to_idx[track_i]

        x1,y1,x2,y2= boxes_i
        c_xi= (x1+x2)/2
        b_yi= y2
        
        h_i= (y2-y1)

        for track_j, boxes_j in person_boxes.items():
            j= id_to_idx[track_j]
            if j<=i:
                continue
            x3,y3,x4,y4= boxes_j

            c_xj= (x3+x4)/2
            b_yj= y4

            h_j= (y4-y3)

            dx= c_xi-c_xj
            dy= b_yi-b_yj

            h_max= max(h_i, h_j)
            h_min= min(h_i, h_j)

            h_ratio= h_max/h_min

            px_dist[(i,j)]=math.sqrt(dx**2 + dy**2)
            h_ratios[(i,j)]= h_ratio
        
    total_boxes= len(person_boxes)
    return px_dist, h_ratios, total_boxes, track_ids, id_to_idx

def get_penalized_dist(px_dist, h_ratios, total_boxes, k_vals):
    # k_vals=[0, 0.3, 0.5, 0.7, 1]
    # k_vals=[0.5]
    complete_penalized_dist_map={}

    for p in range(len(k_vals)):
        
        penalized_dist= np.zeros((total_boxes, total_boxes))
        for key,value in px_dist.items():
            i,j= key

            val1= h_ratios[key]**k_vals[p]
            val2= value

            penalized_dist[i][j]=val1*val2
            penalized_dist[j][i]=penalized_dist[i][j]

        complete_penalized_dist_map[k_vals[p]]=penalized_dist

    return complete_penalized_dist_map,k_vals


if __name__== "__main__":
    person_boxes, result= detect_people('../testImage/test6.png')
    px_dist, h_ratios,total_boxes, track_ids, id_to_idx=extract_metrics(person_boxes)
    print('length of px_dist is :', len(px_dist))
    print("\nPixel Distances:\n")

    for (i, j), dist in sorted(px_dist.items()):
        print(f"{f'P{i} <-> P{j}':<15} | {dist:8.2f} px")

    k_vals= [0.5]
    complete_penalized_map,k_vals=get_penalized_dist(px_dist,h_ratios,total_boxes, k_vals=k_vals)
    print("Complete penalized distance map is: ")
    for k, dist_matrix in complete_penalized_map.items():
        print(f"\n===== k = {k} =====")
        print("      ", end="")
        for j in range(total_boxes):
            print(f"P{j:>8}", end="")
        print()
        for i in range(total_boxes):
            print(f"P{i:<4}", end="")
            for j in range(total_boxes):
                print(f"{dist_matrix[i][j]:8.2f}", end="")
            print()