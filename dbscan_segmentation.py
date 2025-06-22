from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings

warnings.filterwarnings('ignore')

# Load and examine the data
df = pd.read_csv('/Users/rumasinha/Documents/datasets/Mall_Customers.csv')

print("Original Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Features scaling
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data Sample (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=features).head())

# Find optimal eps using k-distance graph
min_samples = 5
neigh = NearestNeighbors(n_neighbors=min_samples)
nbrs = neigh.fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sort distances to k-th nearest neighbor
distances = np.sort(distances[:, min_samples-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points ordered by distance')
plt.ylabel(f'Distance to {min_samples}th nearest neighbor')
plt.title('Knee Method for Optimal Epsilon (DBSCAN)')

# Find knee point
knee = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
plt.axvline(x=knee.knee, color='r', linestyle='--', label=f'Knee at index {knee.knee:.0f}')
plt.legend()
plt.show()

eps_optimal = distances[knee.knee]
print(f"Optimal Epsilon: {eps_optimal:.4f}")

# Perform DBSCAN clustering with Knee Method suggested parameters
dbscan_knee = DBSCAN(eps=eps_optimal, min_samples=5)
clusters_knee = dbscan_knee.fit_predict(X_scaled)

# Add the cluster labels to the original data
df['DBSCAN_Knee_Cluster'] = clusters_knee

# Calculate the mean values for each cluster
dbscan_knee_cluster_means = df.groupby('DBSCAN_Knee_Cluster')[features].mean()

# Display the cluster means
print("\nDBSCAN Cluster Means (Knee Method based):")
print(dbscan_knee_cluster_means)

print(f"\nDBSCAN Knee Method Cluster Distribution:\n{df['DBSCAN_Knee_Cluster'].value_counts()}")

# Calculate Silhouette Score for Knee Method DBSCAN (excluding noise)
X_dbscan_knee_clustered = X_scaled[clusters_knee != -1]
labels_dbscan_knee_clustered = clusters_knee[clusters_knee != -1]

dbscan_knee_silhouette_score = np.nan
if len(np.unique(labels_dbscan_knee_clustered)) >= 2:
    dbscan_knee_silhouette_score = silhouette_score(X_dbscan_knee_clustered, labels_dbscan_knee_clustered)
    print(f"Silhouette Score for DBSCAN (Knee Method, excluding noise): {dbscan_knee_silhouette_score:.4f}")
else:
    print("Cannot calculate Silhouette Score for DBSCAN (Knee Method): Less than 2 valid clusters found (excluding noise).")

# Grid search for optimal DBSCAN parameters
eps_values = np.linspace(0.3, 1.0, num=15)
min_samples_values = range(3, 15)

best_score = -1
best_params = {}
results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        # Calculate silhouette score (excluding noise)
        valid_clusters = clusters[clusters != -1]
        if len(np.unique(valid_clusters)) >= 2:
            score = silhouette_score(X_scaled[clusters != -1], valid_clusters)
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'silhouette_score': score,
                'num_clusters': len(np.unique(valid_clusters)),
                'noise_points': np.sum(clusters == -1)
            })

            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print(f"Best DBSCAN Parameters: {best_params}")
print(f"Best Silhouette Score: {best_score:.4f}")

# Perform DBSCAN clustering with the best parameters from grid search
final_dbscan_model = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
final_dbscan_clusters = final_dbscan_model.fit_predict(X_scaled)
df['DBSCAN_GS_Cluster'] = final_dbscan_clusters

print(f"\nNumber of DBSCAN Grid Search clusters found (including noise -1): {len(np.unique(final_dbscan_clusters))}")
print(f"Number of DBSCAN Grid Search noise points (-1 label): {np.sum(final_dbscan_clusters == -1)}")
print(f"DBSCAN Grid Search Cluster distribution:\n{df['DBSCAN_GS_Cluster'].value_counts()}")

# Calculate the mean values for each cluster from the Grid Search best model
dbscan_gs_cluster_means = df.groupby('DBSCAN_GS_Cluster')[features].mean()
print("\nDBSCAN Cluster Means (Grid Search Best):")
print(dbscan_gs_cluster_means)

# KDE plots for feature distributions by DBSCAN Cluster
print("\n--- KDE Plots of Feature Distributions by DBSCAN Cluster ---")
for feature in features:
    plt.figure(figsize=(10, 6))
    for cluster in sorted(df['DBSCAN_GS_Cluster'].unique()):
        sns.kdeplot(df[df['DBSCAN_GS_Cluster'] == cluster][feature], label=f'Cluster {cluster}', fill=True, alpha=0.3)
    plt.title(f'Distribution of {feature} by DBSCAN_GS_Cluster Cluster')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Box plots for features by DBSCAN_Cluster Cluster
print("\n--- Box Plots of Features by DBSCAN_Cluster Cluster ---")
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='DBSCAN_GS_Cluster', y=feature, palette='viridis')
    plt.title(f'Box Plot of {feature} by DBSCAN_Cluster Cluster')
    plt.show()

# Plot the clusters (Annual Income vs Spending Score with Centroids)
plt.figure(figsize=(10, 6))

centroids = []
for cluster in np.unique(final_dbscan_clusters):
    if cluster == -1:  # Noise points
        cluster_data = df[df['DBSCAN_GS_Cluster'] == cluster]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                    c='gray', label='Noise', s=60, alpha=0.5, marker='.')
    else:  # Regular clusters
        cluster_data = df[df['DBSCAN_GS_Cluster'] == cluster]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                    label=f'Cluster {cluster}', s=60, alpha=0.7)

        # Calculate centroid for this cluster
        center_income = cluster_data['Annual Income (k$)'].mean()
        center_spending = cluster_data['Spending Score (1-100)'].mean()
        centroids.append([center_income, center_spending])

# Plot manually calculated centroids
if centroids:
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                s=300, c='red', marker='X', edgecolors='black',
                label='Manual Centroids', linewidths=2)

plt.title('DBSCAN Clusters with Manual Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
