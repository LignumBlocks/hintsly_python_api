import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator #For elbow method
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log')

def save_plot(fig, filename):
    """Saves a Matplotlib figure to the log directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = os.path.join(LOG_DIR, f"{timestamp}_{filename}.png")
    fig.savefig(filepath)
    plt.close(fig)  # Close the figure to release resources

def plot_kmeans_silhouette(data, labels, n_clusters, silhouette_avg, name):
    """Generates and saves the silhouette plot for KMeans."""
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_samples(data, labels)[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for KMeans (k={n_clusters}, Silhouette Score={silhouette_avg:.2f})")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    save_plot(fig, f"{name}_kmeans_silhouette")

def perform_kmeans(data, n_clusters, pca_components=None, process_name='', log_dir=None):
    """Performs KMeans clustering with optional PCA."""
    if pca_components:
        pca = PCA(n_components=pca_components)
        data = pca.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)  #Set random_state for reproducibility
    kmeans.fit(data)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data, labels)

    if log_dir:  #Save data & plots
        np.save(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_means_labels.npy"), labels)
        plot_kmeans_silhouette(data, labels, n_clusters, silhouette_avg, process_name)
        # if pca_components:
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c=labels) #Assumes 2D data or first 2 components from PCA
        ax.set_title(f"KMeans Clustering (k={n_clusters}, Silhouette Score={silhouette_avg:.2f})")
        save_plot(fig, "kmeans_clusters")
    return labels

def find_optimal_k_kmeans(data, max_k=10, pca_components=None, log_dir=None):
    """Finds the optimal number of clusters for KMeans using the elbow method."""
    inertias = []
    for k in range(1, max_k + 1):
        if pca_components:
            pca = PCA(n_components=pca_components)
            data_pca = pca.fit_transform(data)
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data_pca)
            inertias.append(kmeans.inertia_)
        else:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

    kneedle = KneeLocator(range(1, max_k + 1), inertias, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow

    if log_dir:
        fig, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), inertias, marker='o')
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title(f"Elbow Method for Optimal k ({optimal_k})")
        save_plot(fig, f"kmeans_elbow_k={optimal_k}")

    return optimal_k

def perform_dbscan(data, eps, min_samples, pca_components=None, log_dir=None):
    """Performs DBSCAN clustering with optional PCA."""

    if pca_components:
        pca = PCA(n_components=pca_components)
        data = pca.fit_transform(data)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    # silhouette_avg = silhouette_score(data, labels)

    if log_dir:  #Save data & plots
        np.save(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_dbscan_labels.npy"), labels)
        # if pca_components:
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c=labels) #Assumes 2D data or first 2 components from PCA
        ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        save_plot(fig, "dbscan_clusters")
    return labels

def tune_dbscan_hyperparameters(data, eps_range, min_samples_range, pca_components=None, log_dir=None):
    """Tunes DBSCAN hyperparameters using silhouette score (if possible)."""
    results = {}  # Store results for plotting
    best_silhouette = -1
    best_eps = None
    best_min_samples = None
    best_labels = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = perform_dbscan(data, eps, min_samples, pca_components,log_dir)
            #DBSCAN can produce -1 labels which can cause issues for silhouette score
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                silhouette_avg = silhouette_score(data, labels)
                results[(eps, min_samples)] = silhouette_avg
            else:
                results[(eps, min_samples)] = -1 # Indicate failure

    # Create the plot
    eps_values = np.array(list(results.keys()))[:, 0]
    min_samples_values = np.array(list(results.keys()))[:, 1]
    silhouette_scores = np.array(list(results.values()))

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

    # Use a scatter plot or heatmap depending on your preference
    scatter = ax.scatter(eps_values, min_samples_values, c=silhouette_scores, cmap='viridis', s=100, vmin=-1, vmax=1) #s is size of dots.
    fig.colorbar(scatter, label='Silhouette Score')


    ax.set_xlabel('eps')
    ax.set_ylabel('min_samples')
    ax.set_title('DBSCAN Silhouette Scores')
    plt.xticks(eps_range)
    plt.yticks(min_samples_range)
    save_plot(fig, "dbscan_silhouette_scores")
    plt.show()


    best_silhouette = -1
    best_eps = None
    best_min_samples = None
    best_labels = None
    for (eps, min_samples), score in results.items():
        if score > best_silhouette:
            best_silhouette = score
            best_eps = eps
            best_min_samples = min_samples
            labels = perform_dbscan(data,eps,min_samples,pca_components,log_dir) #Get the labels for best combination.
            best_labels = labels

    return best_eps, best_min_samples, best_labels


# #Generate some sample 2D data (replace with your actual data)
# np.random.seed(0)
# data = np.random.rand(100, 2) * 10  #100 samples, 2 features

# # # KMeans
# # optimal_k = find_optimal_k_kmeans(data, max_k=10, log_dir=LOG_DIR)
# # print(f"Optimal k for KMeans: {optimal_k}")
# # kmeans_labels = perform_kmeans(data, optimal_k, log_dir=LOG_DIR)

# # # KMeans with PCA
# # kmeans_labels_pca = perform_kmeans(data, optimal_k, pca_components='mle', log_dir=LOG_DIR)


# # DBSCAN
# eps_range = np.arange(0.1, 2, 0.2)  # Adjust the range as needed
# min_samples_range = range(2, 7) # Adjust the range as needed
# best_eps, best_min_samples, dbscan_labels = tune_dbscan_hyperparameters(data, eps_range, min_samples_range, log_dir=LOG_DIR)
# print(f"Best eps for DBSCAN: {best_eps}, Best min_samples: {best_min_samples}")

# # DBSCAN with PCA
# best_eps_pca, best_min_samples_pca, dbscan_labels_pca = tune_dbscan_hyperparameters(data, eps_range, min_samples_range, pca_components='mle', log_dir=LOG_DIR)
# print(f"Best eps for DBSCAN (with PCA): {best_eps_pca}, Best min_samples: {best_min_samples_pca}")
