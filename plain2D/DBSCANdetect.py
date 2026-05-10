import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from yoloDetect import detect_people
from penalizedDist import extract_metrics, get_penalized_dist
import cv2
import random

# Fixed colors for groups (expanded to 20)
GROUP_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (192, 192, 192),  # Silver
    (255, 192, 203),  # Pink
    (165, 42, 42),    # Brown
    (75, 0, 130),     # Indigo
    (255, 215, 0),    # Gold
    (0, 100, 0),      # Forest
    (100, 100, 100),  # Gray
]
NOISE_COLOR = (180/255, 180/255, 180/255)  # Gray for noise
TEXT_COLOR = (255, 255, 255)   # White text


def run_dbscan(penalized_dist_map, k_vals, total_boxes, min_samples=3, eps_values=None):
    """
    Run DBSCAN across different k values AND eps values.
    Returns nested dict: result_map[k][eps] = cluster_labels
    """
    if eps_values is None:
        eps_values = [100, 150, 200, 250, 300]
    
    results = {}
    
    for k in k_vals:
        dist_matrix = penalized_dist_map[k]
        results[k] = {}
        
        for eps in eps_values:
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'
            )
            labels = clustering.fit_predict(dist_matrix)
            results[k][eps] = labels
    
    return results, eps_values


def draw_clusters(image_path, person_boxes, labels, k_val, eps_val, save_path=None):
    """
    Draw bounding boxes color-coded by group membership.
    Also labels each person with their group ID.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # for matplotlib
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image)
    
    group_counts = {}
    
    for idx, box in person_boxes.items():
        x1, y1, x2, y2 = box
        label = labels[idx]
        
        if label == -1:
            color = NOISE_COLOR
            group_name = "Noise"
        else:
            color = GROUP_COLORS[label % len(GROUP_COLORS)]
            color = tuple(c / 255.0 for c in color)  # normalize for matplotlib
            group_name = f"Group {label}"
            group_counts[label] = group_counts.get(label, 0) + 1
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(
            x1, y1 - 5, f"P{idx}:{group_name}",
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
        )
    
    # Title with run info
    n_groups = len(group_counts)
    n_noise = sum(1 for l in labels if l == -1)
    group_sizes = list(group_counts.values()) if group_counts else []
    
    title = (
        f"k={k_val}, eps={eps_val} | "
        f"Groups: {n_groups} (sizes: {group_sizes}) | "
        f"Noise: {n_noise}"
    )
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_distance_histogram(px_dist, h_ratios, penalized_dist_map, k_vals):
    """
    Shows the distribution of penalized distances for each k value.
    This helps you see where to set eps:
    - Set eps in the "valley" between noise pairs and group pairs
    """
    fig, axes = plt.subplots(1, len(k_vals), figsize=(5 * len(k_vals), 4))
    
    if len(k_vals) == 1:
        axes = [axes]
    
    for ax, k in zip(axes, k_vals):
        dist_matrix = penalized_dist_map[k]
        # Extract upper triangle (non-zero values)
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        distances = distances[distances > 0]
        
        ax.hist(distances, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.median(distances), color='red', linestyle='--', label=f'Median: {np.median(distances):.0f}')
        ax.axvline(np.percentile(distances, 25), color='orange', linestyle=':', label=f'Q1: {np.percentile(distances, 25):.0f}')
        ax.axvline(np.percentile(distances, 75), color='orange', linestyle=':', label=f'Q3: {np.percentile(distances, 75):.0f}')
        ax.set_title(f'k = {k}')
        ax.set_xlabel('Penalized Distance (pixels)')
        ax.set_ylabel('Pair Count')
        ax.legend(fontsize=8)
    
    plt.suptitle('Penalized Distance Distributions (helps choose eps)')
    plt.tight_layout()
    plt.show()


def plot_pair_scatter(px_dist, h_ratios, penalized_dist_map, k_vals, labels_map, eps_val=200):
    """
    Scatter plot: each pair is a point.
    X = pixel distance, Y = height ratio, Color = penalized distance
    Also annotates which pairs are clustered together at a specific eps.
    """
    # Build pair list
    pairs = []
    for (i, j), dist in px_dist.items():
        pairs.append({
            'i': i, 'j': j,
            'px_dist': dist,
            'h_ratio': h_ratios[(i, j)],
        })
    
    n_k = len(k_vals)
    fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 5))
    
    if n_k == 1:
        axes = [axes]
    
    for ax, k in zip(axes, k_vals):
        x_vals = [p['px_dist'] for p in pairs]
        y_vals = [p['h_ratio'] for p in pairs]
        
        penalized_vals = [penalized_dist_map[k][p['i'], p['j']] for p in pairs]
        
        scatter = ax.scatter(x_vals, y_vals, c=penalized_vals, cmap='plasma', 
                            s=60, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Penalized Dist')
        
        # Draw eps contour line
        if eps_val is not None:
            # The decision boundary for this k: px_dist * h_ratio^k = eps
            # => h_ratio = (eps / px_dist)^(1/k)
            x_line = np.linspace(min(x_vals)*0.9, max(x_vals)*1.1, 200)
            if k > 0:
                y_line = (eps_val / x_line) ** (1/k)
                ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'eps={eps_val}')
            else:
                ax.axvline(eps_val, color='r', linestyle='--', linewidth=2, label=f'eps={eps_val}')
        
        # Label occasional points
        for p in pairs:
            if p['h_ratio'] > 1.5 or p['px_dist'] < 50:  # interesting pairs
                ax.annotate(
                    f"P{p['i']}-P{p['j']}",
                    (p['px_dist'], p['h_ratio']),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points'
                )
        
        ax.set_xlabel('Pixel Distance')
        ax.set_ylabel('Height Ratio')
        ax.set_title(f'k = {k}')
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0.9)  # height_ratio >= 1.0 always
    
    plt.suptitle('Pairwise Plot: Distance vs Height Ratio\n(Red line = clustering boundary)')
    plt.tight_layout()
    plt.show()


def summary_table(results, eps_values, k_vals):
    """
    Print a clean table showing: [k, eps] → n_groups, group_sizes, n_noise
    """
    print("\n" + "=" * 80)
    print("DBSCAN RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'k':<6} {'eps':<6} {'#Groups':<10} {'Group Sizes':<25} {'#Noise':<8}")
    print("-" * 80)
    
    for k in k_vals:
        for eps in eps_values:
            labels = results[k][eps]
            n_noise = sum(1 for l in labels if l == -1)
            
            group_dict = {}
            for l in labels:
                if l != -1:
                    group_dict[l] = group_dict.get(l, 0) + 1
            
            n_groups = len(group_dict)
            sizes = sorted(group_dict.values())
            
            print(f"{k:<6} {eps:<6} {n_groups:<10} {str(sizes):<25} {n_noise:<8}")
        print("-" * 80)


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    IMAGE_PATH = '../testImage/test6.png'
    
    # Step 1: Detect people
    print("Detecting people...")
    person_boxes, result = detect_people(IMAGE_PATH)
    print(f"Found {len(person_boxes)} people\n")
    
    # Step 2: Extract metrics
    print("Extracting pairwise metrics...")
    px_dist, h_ratios, total_boxes = extract_metrics(person_boxes)
    
    # Step 3: Compute penalized distance matrices for k values
    k_vals = [0, 0.3, 0.5, 0.7, 1.0]
    penalized_dist_map, _ = get_penalized_dist(px_dist, h_ratios, total_boxes)
    
    # Step 4: Run DBSCAN across multiple k and eps values
    eps_values = [100, 150, 200, 250, 300, 350, 400]
    results, eps_values = run_dbscan(
        penalized_dist_map, k_vals, total_boxes,
        min_samples=3, eps_values=eps_values
    )
    
    # Step 5: Visualizations
    
    # 5a: Summary table
    summary_table(results, eps_values, k_vals)
    
    # 5b: Distance histograms (helps pick eps range)
    plot_distance_histogram(px_dist, h_ratios, penalized_dist_map, k_vals)
    
    # 5c: Pair scatter plots with decision boundary for eps=200
    plot_pair_scatter(px_dist, h_ratios, penalized_dist_map, k_vals, labels_map=None, eps_val=200)
    
    # 5d: Visualize clusters on the actual image for each (k, eps) combination
    # You can be selective here — maybe just the interesting ones
    critical_combos = [
        (0, 150), (0, 200), (0, 250),
        (0.5, 150), (0.5, 200), (0.5, 250),
        (1.0, 150), (1.0, 200), (1.0, 250),
    ]
    
    for k, eps in critical_combos:
        if k in results and eps in results[k]:
            labels = results[k][eps]
            draw_clusters(
                IMAGE_PATH, person_boxes, labels, k, eps,
                save_path=f'outputImages/test_image6/output_k{k}_eps{eps}.png'
            )