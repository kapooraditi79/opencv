from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



# generate 3d data. guassian normal distributions
X,y= make_blobs(n_samples=47, centers=4,n_features=3, random_state=42, cluster_std=3)
 
# returns a 2d arary with shape (47, 3) and a 1d array with shape (47,)

points= np.array(X)
fig= plt.figure(figsize=(10,10))
ax= fig.add_subplot(111, projection= '3d')
ax.scatter(X[:,0], X[:,1],X[:,2], c= y, cmap='viridis')
plt.show()


# applying the DBSCAN
scaler = StandardScaler()
scaled_points = scaler.fit_transform(points)

dbscan= DBSCAN(eps= 0.5, min_samples= 3)
labels= dbscan.fit_predict(scaled_points)

# analyzing results
# they are labelled as cluster IDs: -1 indicates noise
unique_labels= set(labels)
n_clusters= len(unique_labels)- (1 if -1 in labels else 0)
print(f"Number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
print(f"Noise points: {np.sum(labels == -1)}")

# Combine original points with their cluster labels
clustered_points = np.column_stack((points, labels))
# Display first few entries: [x, y, z, cluster_label]


count=0
for i, p in enumerate(clustered_points):
    if p[3]== -1:
        count+=1
    print(p, p[3])

print(count)

# Assign colors: use a colormap for clusters, red for noise (label -1)
colors = plt.cm.tab10(labels.astype(float) / len(set(labels)))
colors[labels == -1] = [1, 0, 0, 1]  # Red for noise

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=50)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title("DBSCAN Clusters (Red = Noise)")
plt.show()
