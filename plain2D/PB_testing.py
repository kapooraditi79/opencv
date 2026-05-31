# file 9: test_persistent_tracking.py
"""
Test PersistentGroupTracker with video frames.
This simulates processing multiple frames sequentially.
"""

import cv2
import time
import numpy as np
from PBVideoDetect import detect_people
from PBpenalizedDist import extract_metrics, get_penalized_dist
from sklearn.cluster import DBSCAN
from PBgroupPersistance import PersistentGroupTracker 

def process_frame(frame, tracker, current_frame, k_val=0.5, eps_val=100):
    """
    Process a single frame through the entire pipeline.
    
    Args:
        frame: Image frame (numpy array)
        tracker: PersistentGroupTracker instance
        current_frame: Frame number
        k_val: Penalized distance parameter
        eps_val: DBSCAN eps parameter
    
    Returns:
        Dictionary with tracking results and visualization data
    """
    
    # Step 1: Detect people
    person_boxes, result = detect_people(frame)
    start= time.time()
    
    if not person_boxes:
        return {
            "person_groups": {},
            "track_boxes": {},
            "active_groups": {},
            "grace_groups": {},
            "pending_groups": {}
        }
    
    # Step 2: Extract metrics for DBSCAN
    px_dist, h_ratios, total_boxes, track_ids, id_to_idx = extract_metrics(person_boxes)
    
    # Step 3: Compute penalized distance matrix
    penalized_dist_map, _ = get_penalized_dist(px_dist, h_ratios, total_boxes, [k_val])
    dist_matrix = penalized_dist_map[k_val]
    
    # Step 4: Run DBSCAN
    clustering = DBSCAN(eps=eps_val, min_samples=3, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)
    
    # Step 5: Create track_cluster_map for Phase B
    # Map each track_id to its DBSCAN cluster label (-1 for noise)
    track_cluster_map = {}
    for track_id, idx in id_to_idx.items():
        track_cluster_map[track_id] = labels[idx]
    
    # Step 6: Update Persistent Group Tracker
    tracking_result = tracker.update(
        track_cluster_map=track_cluster_map,
        track_boxes=person_boxes,
        current_frame=current_frame
    )
    
    return tracking_result


def draw_tracking_results(frame, tracking_result, frame_number):
    """
    Draw bounding boxes with persistent group colors.
    
    Args:
        frame: Image frame
        tracking_result: Output from PersistentGroupTracker.update()
        frame_number: Frame number for display
    
    Returns:
        Annotated frame
    """
    annotated = frame
    
    # Get data from tracking result
    person_groups = tracking_result["person_groups"]  # track_id → group_id or None
    active_groups = tracking_result["active_groups"]  # group_id → display info
    grace_groups = tracking_result["grace_groups"]    # group_id → fading info
    pending_groups = tracking_result["pending_groups"] # group_id → pending info
    track_boxes = tracking_result["track_boxes"]      # track_id → [x1,y1,x2,y2]
    
    # Create reverse mapping: group_id → color
    group_colors = {}
    for gid, info in active_groups.items():
        group_colors[gid] = info["color"]
    for gid, info in grace_groups.items():
        # For grace groups, apply fading
        color = info["color"]
        opacity = info["opacity"]
        group_colors[gid] = tuple(int(c * opacity) for c in color)
    
    # Draw each person
    for track_id, box in track_boxes.items():
        x1, y1, x2, y2 = [int(coord) for coord in box]
        group_id = person_groups.get(track_id)
        
        # Determine color and style
        if group_id is None:
            # Noise - gray
            color = (180, 180, 180)
            thickness = 1
            linetype = cv2.LINE_AA
            label = "Noise"
        elif group_id in active_groups:
            # Active confirmed group
            color = group_colors[group_id]
            thickness = 2
            linetype = cv2.LINE_AA
            label = f"G{group_id}({active_groups[group_id]['member_count']})"
        elif group_id in grace_groups:
            # Grace period (fading)
            color = group_colors[group_id]
            thickness = 2
            linetype = cv2.LINE_AA
            label = f"G{group_id}(fading)"
        elif group_id in pending_groups:
            # Pending group - dashed border
            color = (128, 128, 128)
            thickness = 1
            linetype = cv2.LINE_AA
            label = "?"
        else:
            # Fallback
            color = (100, 100, 100)
            thickness = 1
            linetype = cv2.LINE_AA
            label = f"G{group_id}"
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, linetype)
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add frame info
    cv2.putText(annotated, f"Frame: {frame_number}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add group statistics
    y_offset = 60
    cv2.putText(annotated, f"Active Groups: {len(active_groups)}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(annotated, f"Grace Groups: {len(grace_groups)}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(annotated, f"Pending Groups: {len(pending_groups)}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated


# Test with a video file or image sequence
if __name__ == "__main__":
    # For testing with a single image (simulate multiple frames)
    IMAGE_PATH = '../testImage/test6.png'
    VIDEO_PATH = '../testVideo/test2.mp4'  # Replace with actual video path if available
    
    # Initialize tracker (assuming 30 fps for image test)
    tracker = PersistentGroupTracker(fps=30)
    
    if VIDEO_PATH:
        # Process video
        cap = cv2.VideoCapture(VIDEO_PATH)
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % 3 != 0:
                frame_num += 1
                continue
            # frame = cv2.resize(frame, (960, 540))
            
            # Process frame
            tracking_result = process_frame(frame, tracker, frame_num, k_val=0.5, eps_val=100)
            
            # Draw results
            annotated = draw_tracking_results(frame, tracking_result, frame_num)
            
            # Display
            cv2.imshow('Persistent Group Tracking', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_num += 1
        
        cap.release()
        cv2.destroyAllWindows()

        print("\n\n")
        print("=" * 60)
        print("GROUP HISTORY REPORT")
        print("=" * 60)
        
        # Option 1: Pretty-print all groups
        tracker.print_history()
    
    else:
        # Test with static image - simulate multiple frames by reusing same image
        print("Testing with static image (simulating 10 frames)...")
        frame = cv2.imread(IMAGE_PATH)
        
        for frame_num in range(10):
            print(f"\n--- Frame {frame_num} ---")
            
            # Process same image each time
            tracking_result = process_frame(frame, tracker, frame_num, k_val=0.5, eps_val=100)
            
            # Print events
            if tracking_result.get("events"):
                for event in tracking_result["events"]:
                    print(f"  Event: {event}")
            
            # Draw and save frame
            annotated = draw_tracking_results(frame, tracking_result, frame_num)
            cv2.imwrite(f'test_output/frame_{frame_num:03d}.png', annotated)
            
            # Display
            cv2.imshow('Persistent Group Tracking', annotated)
            cv2.waitKey(500)  # Wait 500ms between frames
        
        cv2.destroyAllWindows()
        print("\nTest complete! Check test_output/ for results.")