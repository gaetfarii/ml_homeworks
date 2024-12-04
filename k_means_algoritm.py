import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


dataset_iris = load_iris()
array_iris = dataset_iris.data
clusters_count = 3  # Количество кластеров
max_iter = 100
images = []
np.random.seed(0)
centroids = array_iris[np.random.choice(range(len(array_iris)), clusters_count, replace=False)]


def metrics(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def appropriation_point_clusters(array_iris, centroids):
    clusters = []
    for point in array_iris:
        distances = [metrics(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # находим и добавляем индексы минимальных расстояний в массив
        clusters.append(cluster)
    return np.array(clusters)


def update_centroids(array_iris, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = array_iris[clusters == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


for iteration in range(max_iter):
    clusters = appropriation_point_clusters(array_iris, centroids)
    plt.figure(figsize=(6, 6))
    colors = ['orange', 'g', 'b']

    for i in range(clusters_count):
        cluster_points = array_iris[clusters == i]  # принадлежность точки кластеру
        plt.scatter(cluster_points[:, 0], cluster_points[:,1], c=colors[i], label=f'Кластер {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Центроиды')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'K-means Clustering - Итерация {iteration + 1}')
    plt.legend()

    plt.savefig(f'iteration_{iteration}.png')
    plt.close()

    new_centroids = update_centroids(array_iris, clusters, clusters_count)

    if np.all(centroids == new_centroids):
        print("Отработал", iteration + 1)
        break

    centroids = new_centroids
