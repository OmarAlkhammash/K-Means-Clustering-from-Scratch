import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification

class KMeansClustering:
    def __init__(self, dataset_type='blobs', n_samples=300, n_clusters=3, max_iters=100, seed=42):
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.seed = seed
        self.centroids = None
        self.labels = None
        self.X, self.y = self.load_data()

    def load_data(self):
        dataset_options = {
            'blobs': make_blobs,
            'moons': make_moons,
            'circles': make_circles,
            'classification': make_classification
        }

        dataset_func = dataset_options[self.dataset_type]
        
        if self.dataset_type == 'blobs':
            return dataset_func(n_samples=self.n_samples, centers=self.n_clusters, random_state=self.seed)
        elif self.dataset_type == 'moons':
            return dataset_func(n_samples=self.n_samples, noise=0.1, random_state=self.seed)
        elif self.dataset_type == 'circles':
            return dataset_func(n_samples=self.n_samples, noise=0.1, factor=0.5, random_state=self.seed)
        elif self.dataset_type == 'classification':
            return dataset_func(n_samples=self.n_samples, n_features=2, n_informative=2, n_redundant=0, n_classes=self.n_clusters, n_clusters_per_class=1, random_state=self.seed)

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def fit(self):
        np.random.seed(self.seed)
        self.centroids = self.X[np.random.choice(self.X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.n_clusters)]
            for point in self.X:
                distances = self.euclidean_distance(point, self.centroids)
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)

            new_centroids = np.array([np.mean(cluster, axis=0) if cluster else self.centroids[i] for i, cluster in enumerate(clusters)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels = np.zeros(self.X.shape[0])
        for i, cluster in enumerate(clusters):
            for point in cluster:
                self.labels[np.where((self.X == point).all(axis=1))] = i

    def plot(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
        plt.title(f'K-Means: {self.dataset_type.capitalize()}')
        plt.show()

datasets = ['blobs', 'moons', 'circles', 'classification']

for dataset in datasets:
    print(f"K-Means: {dataset.capitalize()}")
    kmeans = KMeansClustering(dataset_type=dataset, n_samples=300, n_clusters=3)
    kmeans.fit()
    kmeans.plot()