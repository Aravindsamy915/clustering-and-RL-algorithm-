import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Sample Customer Segmentation Data
data = {
    "CustomerID": range(1, 21),
    "Age": [23, 45, 34, 29, 40, 23, 50, 42, 35, 28, 36, 52, 41, 30, 24, 60, 38, 25, 27, 48],
    "Annual Income": [45000, 54000, 60000, 52000, 58000, 75000, 80000, 52000, 62000, 64000, 
                      72000, 90000, 48000, 50000, 52000, 105000, 75000, 68000, 49000, 91000],
    "Spending Score": [39, 81, 6, 77, 40, 76, 94, 3, 72, 14, 99, 15, 39, 82, 96, 12, 77, 13, 98, 6]
}

df = pd.DataFrame(data)

# Standardize the data
X = df[['Age', 'Annual Income', 'Spending Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a mesh for plotting decision boundaries with the first two features
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Add a constant value for the third feature (Spending Score)
spending_score_mean = X_scaled[:, 2].mean()
grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, spending_score_mean)]

### 1. Gaussian Mixture Model (GMM) Clustering
try:
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_scaled)
    
    # Initial clusters based on GMM before fitting
    initial_gmm_labels = gmm.predict(X_scaled)

    # Final clusters after fitting
    gmm_labels = gmm.predict(X_scaled)
    gmm_grid_labels = gmm.predict(grid).reshape(xx.shape)

    # Display the number of epochs (iterations for convergence)
    epochs_gmm = gmm.n_iter_

    # Calculate silhouette score for GMM as error rate
    silhouette_avg_gmm = silhouette_score(X_scaled, gmm_labels)
    
except Exception as e:
    print("GMM Clustering Error:", e)
    initial_gmm_labels, gmm_labels, epochs_gmm, silhouette_avg_gmm = None, None, None, None

### 2. Spectral Clustering
try:
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
    
    # Random initial labels for Spectral Clustering
    initial_spectral_labels = np.random.randint(0, 3, X_scaled.shape[0])
    
    # Final clusters after fitting
    spectral_labels = spectral.fit_predict(X_scaled)

    # Silhouette score for Spectral as error rate
    silhouette_avg_spectral = silhouette_score(X_scaled, spectral_labels)

except Exception as e:
    print("Spectral Clustering Error:", e)
    initial_spectral_labels, spectral_labels, silhouette_avg_spectral = None, None, None

# Plotting
plt.figure(figsize=(14, 6))

# GMM Plot with a contour plot for boundaries
if gmm_labels is not None:
    plt.subplot(1, 2, 1)
    cmap_gmm = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, gmm_grid_labels, cmap=cmap_gmm, alpha=0.3)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis', edgecolor='k', s=50)
    plt.title("GMM Clustering")
    plt.xlabel('Age (scaled)')
    plt.ylabel('Annual Income (scaled)')

# Spectral Clustering Plot with a scatter style for boundaries
if spectral_labels is not None:
    plt.subplot(1, 2, 2)
    cmap_spectral = ListedColormap(['#AAF0D1', '#F0AFAF', '#AFAFF0'])
    plt.scatter(xx.ravel(), yy.ravel(), c=spectral.fit_predict(grid), cmap=cmap_spectral, alpha=0.03)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=spectral_labels, cmap='plasma', edgecolor='k', s=50)
    plt.title("Spectral Clustering")
    plt.xlabel('Age (scaled)')
    plt.ylabel('Annual Income (scaled)')

plt.tight_layout()
plt.show()

# Display Initial and Final Cluster Details, Error Rate, and Epochs
print("=== Gaussian Mixture Model (GMM) ===")
print("Initial Clusters (GMM):", initial_gmm_labels)
print("Final Clusters (GMM):", gmm_labels)
print("Convergence Epochs (GMM):", epochs_gmm)
print("Silhouette Score (GMM) as Error Rate:", silhouette_avg_gmm)

print("\n=== Spectral Clustering ===")
print("Initial Clusters (Random Start for Spectral):", initial_spectral_labels)
print("Final Clusters (Spectral):", spectral_labels)
print("Silhouette Score (Spectral) as Error Rate:", silhouette_avg_spectral)
