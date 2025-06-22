from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import hdbscan
from kneed import KneeLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
import matplotlib.cm as cm

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

# HDBSCAN parameter optimization
param_grid = {
    'min_cluster_size': [5, 10, 15, 20],
    'min_samples': [None, 3, 5, 10]
}

best_score = -1
best_params = {}
results = []

for min_cluster_size in param_grid['min_cluster_size']:
    for min_samples in param_grid['min_samples']:
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

        clusters = hdbscan_model.fit_predict(X_scaled)
        valid_clusters = clusters[clusters != -1]

        if len(np.unique(valid_clusters)) >= 2:
            score = silhouette_score(X_scaled[clusters != -1], valid_clusters)
            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'silhouette_score': score,
                'num_clusters': len(np.unique(valid_clusters))
            })

            if score > best_score:
                best_score = score
                best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}

print(f"Best HDBSCAN Parameters: {best_params}")
print(f"Best Silhouette Score: {best_score:.4f}")

print("\n--- Re-running HDBSCAN with best parameters to get final clusters ---")
final_hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=best_params['min_cluster_size'],
                                          min_samples=best_params['min_samples'] if best_params['min_samples'] is not None else best_params['min_cluster_size'],
                                          prediction_data=True)
final_hdbscan_clusters = final_hdbscan_model.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
df['HDBSCAN_Cluster'] = final_hdbscan_clusters

print(f"\nNumber of HDBSCAN clusters found (including noise -1): {len(np.unique(final_hdbscan_clusters))}")
print(f"Number of HDBSCAN noise points (-1 label): {np.sum(final_hdbscan_clusters == -1)}")
print(f"HDBSCAN Cluster distribution:\n{df['HDBSCAN_Cluster'].value_counts()}")

# --- Explicitly calculate and display the Silhouette Score for the FINAL HDBSCAN clusters ---
X_hdbscan_clustered = X_scaled[final_hdbscan_clusters != -1]
labels_hdbscan_clustered = final_hdbscan_clusters[final_hdbscan_clusters != -1]

final_hdbscan_silhouette_score = np.nan
if len(np.unique(labels_hdbscan_clustered)) >= 2:
    final_hdbscan_silhouette_score = silhouette_score(X_hdbscan_clustered, labels_hdbscan_clustered)
    print(f"\nSilhouette Score for the FINAL HDBSCAN clusters: {final_hdbscan_silhouette_score:.4f}")
else:
    print("\nCannot calculate Silhouette Score for final HDBSCAN clusters: Less than 2 valid clusters found (excluding noise).")

# Calculate the mean values for each cluster from the Grid Search best model
hdbscan_cluster_means = df.groupby('HDBSCAN_Cluster')[features].mean()
print("\nHDBSCAN Cluster Means:")
print(hdbscan_cluster_means)

# KDE plots for feature distributions by HDBSCAN Cluster
print("\n--- KDE Plots of Feature Distributions by HDBSACN Cluster ---")
for feature in features:
    plt.figure(figsize=(10, 6))
    for cluster in sorted(df['HDBSCAN_Cluster'].unique()):
        sns.kdeplot(df[df['HDBSCAN_Cluster'] == cluster][feature], label=f'Cluster {cluster}', fill=True, alpha=0.3)
    plt.title(f'Distribution of {feature} by HDBSCAN_Cluster Cluster')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Box plots for features by HDBSCAN_Cluster Cluster
print("\n--- Box Plots of Features by HDBSCAN_Cluster Cluster ---")
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='HDBSCAN_Cluster', y=feature, palette='viridis')
    plt.title(f'Box Plot of {feature} by HDBSCAN_Cluster Cluster')
    plt.show()

# Plot the clusters (Annual Income vs Spending Score with Centroids)
plt.figure(figsize=(10, 6))

centroids = []
for cluster in np.unique(final_hdbscan_clusters):
    if cluster == -1:  # Noise points
        cluster_data = df[df['HDBSCAN_Cluster'] == cluster]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                    c='gray', label='Noise', s=60, alpha=0.5, marker='.')
    else:  # Regular clusters
        cluster_data = df[df['HDBSCAN_Cluster'] == cluster]
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

plt.title('HDBSCAN Clusters with Manual Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()