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
df = pd.read_csv('Mall_Customers.csv')

print("Original Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

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

# Features scaling
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data Sample (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=features).head())

# Determining Optimal k: The Elbow Method
# Determine optimal number of clusters using Elbow Method
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('K-Means Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Perform K-Means with optimal clusters
optimal_kmeans_clusters = 5
kmeans_model = KMeans(n_clusters=optimal_kmeans_clusters, init='k-means++',
                      random_state=42, n_init=10)
kmeans_clusters = kmeans_model.fit_predict(X_scaled)

# Evaluate clustering quality
kmeans_silhouette_avg = silhouette_score(X_scaled, kmeans_clusters)
print(f"K-Means Silhouette Score (k={optimal_kmeans_clusters}): {kmeans_silhouette_avg:.4f}")

# Add cluster labels to original data
df['KMeans_Cluster'] = kmeans_clusters

# Calculate cluster characteristics
kmeans_cluster_means = df.groupby('KMeans_Cluster')[features].mean()
print("\nK-Means Cluster Characteristics:")
print(kmeans_cluster_means)
print(f"\nCluster Distribution:\n{df['KMeans_Cluster'].value_counts()}")


# Plot the clusters (Annual Income vs Spending Score with Centroids)
plt.figure(figsize=(10, 6))
for cluster in range(optimal_kmeans_clusters):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {cluster}', s=60, alpha=0.7)

# Plot centroids
plt.scatter(kmeans_model.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1], # Inverse transform for original scale
            kmeans_model.cluster_centers_[:, 2] * scaler.scale_[2] + scaler.mean_[2],
            s=300, c='red', marker='X', edgecolors='black', label='Centroids')
plt.title('K-Means Clusters of Mall Customers (Income vs. Spending)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Silhouette Plot for K-Means
print("\n--- Silhouette Analysis for K-Means Clusters ---")
range_n_clusters = [optimal_kmeans_clusters] # Just plot for the chosen optimal k

for n_clusters in range_n_clusters: # Loop will run once for the optimal_kmeans_clusters
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot for K-Means ({n_clusters} clusters)")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

# Box plots for features by K-Means Cluster
print("\n--- Box Plots of Features by K-Means Cluster ---")
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='KMeans_Cluster', y=feature, palette='viridis')
    plt.title(f'Box Plot of {feature} by K-Means Cluster')
    plt.show()

# Heatmap of K-Means Cluster Centers (Scaled Features)
print("\n--- Heatmap of K-Means Cluster Centers (Scaled) ---")
# To plot the heatmap with original scale features, inverse transform the centers
kmeans_cluster_centers_original_scale = scaler.inverse_transform(kmeans_model.cluster_centers_)
cluster_centers_df = pd.DataFrame(kmeans_cluster_centers_original_scale, columns=features)
plt.figure(figsize=(10, 7))
sns.heatmap(cluster_centers_df, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Heatmap of K-Means Cluster Centers (Original Scale)')
plt.xlabel('Features')
plt.ylabel('Cluster ID')
plt.show()

# KDE plots for feature distributions by K-Means Cluster
print("\n--- KDE Plots of Feature Distributions by K-Means Cluster ---")
for feature in features:
    plt.figure(figsize=(10, 6))
    for cluster in sorted(df['KMeans_Cluster'].unique()):
        sns.kdeplot(df[df['KMeans_Cluster'] == cluster][feature], label=f'Cluster {cluster}', fill=True, alpha=0.3)
    plt.title(f'Distribution of {feature} by K-Means Cluster')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

