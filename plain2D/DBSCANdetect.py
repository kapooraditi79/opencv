import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import sys
sys.path.append(r'D:\Antares\plain2D')
from penalizedDist import extract_metrics, get_penalized_dist
from yoloDetect import detect_people

def DBSCANcluster(complete_penalized_map, k_vals):
    
