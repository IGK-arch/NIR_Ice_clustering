import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt


def cluster_data(data, n_clusters=5, cluster_type='KMeans', eps=0.5, min_samples=5):
    S, T, H, W = data.shape
    flattened_data = np.array([year_data.ravel() for year_data in data])

    if cluster_type == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=1000)
        labels = model.fit_predict(flattened_data)
    elif cluster_type == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(flattened_data)
    else:
        raise ValueError("Unsupported clustering type. Use 'KMeans' or 'DBSCAN'.")

    return labels, model


def visualize_clusters(data, labels, n_clusters):
    cluster_means = []
    for cluster in range(n_clusters):
        cluster_data_dop = data[labels == cluster]
        mean_data = cluster_data_dop.mean(axis=0)
        cluster_means.append(mean_data)

    fig, axs = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    for cluster in range(n_clusters):
        ax = axs[cluster] if n_clusters > 1 else axs
        ax.imshow(cluster_means[cluster].mean(axis=0), cmap='Blues')
        ax.set_title(f'Cluster {cluster}')
        ax.axis('off')
    plt.show()
