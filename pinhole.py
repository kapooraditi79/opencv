from PeopleDetection import detect_people
from depthpro import get_depth_image, get_feet_depth
import numpy as np 

def pixel_to_3d(x_pixel, y_pixel, depth, fx, fy, cx, cy):
    X = (x_pixel - cx) * depth / fx
    Y = (y_pixel - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def get_3d_coordinates(person_boxes, feet_depths, focallength, depth):
    h, w= depth.shape
    fx= fy= focallength
    cx= w/2
    cy= h/2
    position_3d={}

    for person_id, feet_depth in feet_depths.items():
        x1, y1, x2, y2= person_boxes[person_id]
        feet_x= int((x1+x2)/2)
        feet_y= int(y2)

        coords= pixel_to_3d(feet_x, feet_y, feet_depth,fx,fy,cx,cy)
        position_3d[person_id]= coords
    
    return position_3d

if __name__=='__main__':
    depth, focallength= get_depth_image('testImage/test6.png')
    person_boxes= detect_people('testImage/test6.png')
    feet_depths= get_feet_depth(depth, person_boxes)
    positions_3d= get_3d_coordinates(person_boxes,feet_depths,focallength, depth)
    for person_id, coords in positions_3d.items():
        print(f"Person {person_id} → X: {coords[0]:.2f}m, Y: {coords[1]:.2f}m, Z: {coords[2]:.2f}m")



        

        
