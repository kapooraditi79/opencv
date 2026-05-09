import math
import sys
sys.path.append(r'D:\Antares\plain2D')
import cv2
from yoloDetect import detect_people
from ultralytics import YOLO
import numpy as np

model= YOLO('../yolov10s.pt')

def extract_metrics(person_boxes):
    px_dist= {}
    h_ratios={}
    # building dicts
    for i, boxes_i in person_boxes.items():
        x1,y1,x2,y2= boxes_i

        c_xi= (x1+x2)/2
        b_yi= y2
        
        h_i= (y2-y1)
        w_i= x2-x1

         
        for j, boxes_j in person_boxes.items():
            if j<=i:
                continue
            x3,y3,x4,y4= boxes_j

            c_xj= (x3+x4)/2
            b_yj= y4

            h_j= (y4-y3)
            w_j= x4-x3

            dx= c_xi-c_xj
            dy= b_yi-b_yj

            h_max= max(h_i, h_j)
            h_min= min(h_i, h_j)

            h_ratio= h_max/h_min

            px_dist[(i,j)]=math.sqrt(dx**2 + dy**2)
            h_ratios[(i,j)]= h_ratio
    
    total_boxes= len(person_boxes)
    return px_dist, h_ratios, total_boxes


def get_penalized_dist(px_dist, h_ratios, total_boxes):
    k_vals=[0, 0.3, 0.5, 0.7, 1]
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

    return complete_penalized_dist_map


if __name__=="__main__":
    person_boxes, result= detect_people('../testImage/test6.png')
    px_dist, h_ratios,total_boxes=extract_metrics(person_boxes)
    complete_penalized_map=get_penalized_dist(px_dist,h_ratios,total_boxes)
    print("Complete penalized distance map is: ")
    for k, dist_matrix in complete_penalized_map.items():
        print(f"\n===== k = {k} =====")
        print(np.round(dist_matrix, 3))


