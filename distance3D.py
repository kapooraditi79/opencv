import sys
sys.path.append(r'D:\Antares')  
from pinhole import get_3d_coordinates
from PeopleDetection import detect_people
from depthpro import get_depth_image, get_feet_depth
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN


def get_distance(position_3d):
    # extract the ids and the coords in dict
    ids= list(position_3d.keys())
    coords= np.array([position_3d[i] for i in ids])

    # finding pairwise euclidean distance
    dist_matrix= euclidean_distances(coords, coords)

    # cluster
    clusters= DBSCAN(eps=0.8, min_samples=3,metric= 'precomputed')
    labels= clusters.fit_predict(dist_matrix)  
    labels_dict = {ids[i]: labels[i] for i in range(len(ids))}

    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            dist= dist_matrix[i][j]
            print(f"Person {ids[i]} <-> Person {ids[j]}: {dist:.2f}m")
    
    return dist_matrix, ids, labels_dict


# if __name__=='__main__':
#     depth, focallength=get_depth_image('testImage/test6.png')
#     person_boxes= detect_people('testImage/test6.png')
#     feet_depths= get_feet_depth(depth, person_boxes)
#     position_3d= get_3d_coordinates(person_boxes, feet_depths, focallength, depth)
#     dist_matrix, ids, labels= get_distance(position_3d)
    
    
