import depth_pro
from PIL import Image       # for image handling
import torch
import matplotlib.pyplot as plt
from depth_pro import create_model_and_transforms
import numpy as np

# we will use gpu
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model and preprocessing transform
model, transform= depth_pro.create_model_and_transforms()
# move model to gpu
model.to(device)

# turn on eval
# to turn off dropout, training randomness
model.eval()

# load the image
image, _, f_px = depth_pro.load_rgb('testImage/test6.png')
# f_px: focal length of camera in pixel units
# for converting depth → real-world scale

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

# showing depth map
plt.figure(figsize=(10,5))
plt.title("Depth Map (meters)")
plt.imshow(depth, cmap='inferno')   # red = close, blue = far
plt.colorbar(label="Depth (m)")
plt.show()

# seeing the depth array
print("shape of depth array: ", depth.shape)    
# rows len-> height of the image [in px], x= 1225 [example]
# columns len-> width of image [in px], y= 593 [example]
print("Min depth:", np.min(depth))
print("Max depth:", np.max(depth))
print("Mean depth:", np.mean(depth))

# for some specific pixel
y, x = 400, 500
print(f"Depth at ({x},{y}):", depth[y][x], "meters")


