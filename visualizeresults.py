import sys
sys.path.append(r'D:\Antares')  
import matplotlib.pyplot as plt
import cv2
from pinhole import get_3d_coordinates
from PeopleDetection import detect_people
from depthpro import get_depth_image, get_feet_depth
from distance3D import get_distance
import numpy as np

def visualize(labels, person_boxes, image_path):
    colors= [(0,255,0), (255,0,0), (0,0,255), (255,255,0), 
          (255,0,255), (0,255,255)]
    image_cv= cv2.imread(image_path)

    for person_id, label in labels.items():
        # convert each val to an integer, as px indices need to be ints
        x1,y1,x2,y2= map(int, person_boxes[person_id])  

        if label== -1:
            color= (128,128,128)
        else:
            color= colors[label%len(colors)]

        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_cv, f"G{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Grouped People', image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output_labeled.jpg', image_cv)


if __name__ == '__main__':
    image_path = 'testImage/test6.png'
    depth, focallength = get_depth_image(image_path)
    person_boxes = detect_people(image_path)
    feet_depths = get_feet_depth(depth, person_boxes)
    position_3d = get_3d_coordinates(person_boxes, feet_depths, focallength, depth)
    dist_matrix, ids, labels = get_distance(position_3d)
    visualize(labels, person_boxes, image_path)