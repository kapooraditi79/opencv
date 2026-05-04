import cv2
from ultralytics import YOLO

# get the model
model= YOLO("yolov8n.pt")

image= cv2.imread("image.png")
result= model(image)

annotated_image= result[0].plot()

cv2.imshow("Annotated Image", annotated_image)


cv2.waitKey(0)
cv2.destroyAllWindows() 