import sys
sys.path.append(r'D:\Antares')  
import depth_pro
from PIL import Image       # for image handling
import torch 
import matplotlib.pyplot as plt
from depth_pro import create_model_and_transforms
import numpy as np
from PeopleDetection import detect_people

# we will use gpu
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load once at module level
model, transform = depth_pro.create_model_and_transforms()
model.to(device)
model.eval()


def get_depth_image(image_path):
    image, _, f_px = depth_pro.load_rgb(image_path)
    # f_px: focal length of camera in pixel units
    # for converting depth → real-world scale``
    # now transform
    image = transform(image).to(device)
    # Run inference.
    # feeds image into model
    # uses f_px for scale-aware depth
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    # extract depth
    depth = prediction["depth"].cpu().numpy()  # Depth in [m].   # move the depth to cpu
    # each pixel ->  depth[y][x] = distance from camera (in meters)
    focallength_px = prediction["focallength_px"].cpu().numpy()  # Focal length in pixels.
    return depth, focallength_px


# getting the feet depth
def get_feet_depth(depth_map, person_boxes, region_size=5):
    feet_depth_list= {}
    for i in person_boxes:
        x1,y1, x2, y2= person_boxes[i]

        # feet position
        feet_x= int((x1+x2)/2)
        feet_y= int(y2)  

        # extract the small region around feet
        half= region_size//2
        h,w= depth_map.shape    
        y_start= max(0, feet_y- half)
        y_end= min(h, feet_y+ half+1)
        x_start= max(0, feet_x - half)
        x_end= min(w, feet_x+ half+1)
        
        region= depth_map[y_start:y_end, x_start:x_end]
        feet_depth_list[i]= np.median(region)

    return feet_depth_list


if __name__== '__main__':
    depth, focallength_px= get_depth_image('testImage/test6.png')
    person_boxes= detect_people('testImage/test6.png')
    print("Focal length is: ", focallength_px)
    # showing depth map
    plt.figure(figsize=(10,5))
    plt.title("Depth Map (meters)")
    plt.imshow(depth, cmap='inferno')   # red = close, blue = far
    plt.colorbar(label="Depth (m)")
    plt.show()
    print("shape of depth array: ", depth.shape)    
    # rows len-> height of the image [in px], x= 1225 [example]
    # columns len-> width of image [in px], y= 593 [example]
    print("Min depth:", np.min(depth))
    print("Max depth:", np.max(depth))
    print("Mean depth:", np.mean(depth))

    feet_depths= get_feet_depth(depth, person_boxes)
    print("Feet depths: ", feet_depths) 