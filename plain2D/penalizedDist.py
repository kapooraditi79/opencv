import math
import cv2
from yoloDetect import detect_people
from ultralytics import YOLO

model= YOLO('../yolov10s.pt')

dist= {}
h_ratios={}
def extract_metrics(person_boxes):
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

            dist[(i,j)]=math.sqrt(dx**2 + dy**2)
            h_ratios[(i,j)]= h_ratio

    return dist, h_ratios


def get_penalized_dist()