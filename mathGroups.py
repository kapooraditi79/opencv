import numpy as np
from sklearn.cluster import DBSCAN

def simulate_park_logic():
    # --- SIMULATED DATA ---
    # format: {'id': ID, 'centroid': (x, y), 'height_px': h}
    
    # GROUP 1: Close to camera (Big boxes)
    # They are 200 pixels apart, but 400 pixels tall.
    group_close = [
    {'id': 1, 'centroid': (200, 650), 'height_px': 320},  # sitting
    {'id': 2, 'centroid': (260, 660), 'height_px': 340},  # sitting
    {'id': 3, 'centroid': (320, 645), 'height_px': 360},  # standing
    {'id': 4, 'centroid': (380, 655), 'height_px': 330}
    ]

    group_mid = [
    {'id': 5, 'centroid': (600, 500), 'height_px': 220},
    {'id': 6, 'centroid': (650, 510), 'height_px': 210},
    {'id': 7, 'centroid': (700, 495), 'height_px': 230}
    ]
    
    # GROUP 2: Far from camera (Tiny boxes)
    # They are only 20 pixels apart, and 40 pixels tall.
    group_far = [
    {'id': 8, 'centroid': (750, 200), 'height_px': 60},
    {'id': 9, 'centroid': (780, 210), 'height_px': 55},
    {'id': 10, 'centroid': (770, 230), 'height_px': 65}
    ]
    individuals = [
    {'id': 11, 'centroid': (500, 600), 'height_px': 300},  # close standing
    {'id': 12, 'centroid': (850, 550), 'height_px': 250},  # mid
    {'id': 13, 'centroid': (150, 400), 'height_px': 180},  # mid-far
    {'id': 14, 'centroid': (900, 250), 'height_px': 80}    # far
]
    dog_walkers = [
        {'id': 15, 'centroid': (550, 620), 'height_px': 290},
        {'id': 16, 'centroid': (580, 625), 'height_px': 60}  # dog (small box)
    ]
    
    all_people = group_close + group_far + group_mid + individuals + dog_walkers
    n = len(all_people)
    dist_matrix = np.zeros((n, n))

    print(f"{'Pair':<15} | {'Pixel Dist':<12} | {'Real Dist (m)':<15}")
    print("-" * 50)

    # --- THE MATH ENGINE ---
    for i in range(n):
        for j in range(i + 1, n):
            p1 = all_people[i]
            p2 = all_people[j]
            
            # 1. Get pixel distance
            pix_dist = np.sqrt((p1['centroid'][0] - p2['centroid'][0])**2 + 
                               (p1['centroid'][1] - p2['centroid'][1])**2)
            
            # 2. Heuristic Scaling (The Adjustable Ruler)
            avg_h = (p1['height_px'] + p2['height_px']) / 2
            scale = 1.7 / avg_h
            real_dist = pix_dist * scale
            
            dist_matrix[i, j] = dist_matrix[j, i] = real_dist
            
            # Print only within-group distances to show the logic working
            if (p1['id'] <= 3 and p2['id'] <= 3) or (p1['id'] > 3 and p2['id'] > 3):
                print(f"ID {p1['id']} - ID {p2['id']:<4} | {pix_dist:<12.1f} | {real_dist:<15.2f}")

    # --- CLUSTERING ---
    # eps=2.0 meters, min_samples=3 people 
    db = DBSCAN(eps=1.2, min_samples=3, metric='precomputed').fit(dist_matrix)
    
    print("\n--- RESULTS ---")
    for i, label in enumerate(db.labels_):
        status = f"Group {label}" if label != -1 else "No Group"
        print(f"Person {all_people[i]['id']}: {status}")

simulate_park_logic()