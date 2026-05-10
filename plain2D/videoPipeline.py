import cv2
from ultralytics import YOLO
import sys
sys.path.append(r'D:\Antares\plain2D')
from VidyoloDetect import detect_people
from penalizedDist import extract_metrics, get_penalized_dist

video_path= r'..\testVideo\test1.mp4'
output_path=r'\VidOutput\{video_path}'

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

