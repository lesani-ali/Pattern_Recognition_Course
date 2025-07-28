import numpy as np


class KMeans():
    def __init__(self, n_clusters, max_iters=100, similarity="cosine"):
        """
        Initialize the KMeans algorithm.

        Args:
            n_clusters (int): Number of clusters.
            max_iters (int): Maximum number of iterations.
            similarity (str): Similarity method ('euclidean' or 'cosine').
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.similarity = similarity # "euclidean" or "cosine"
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X):
        """
        Fit the KMeans algorithm to the data.

        Args:
            X (array-like): Input data with shape (n_samples, n_features).
        """
        # Initialize labels randomly
        self.X = X
        self.labels_ = np.random.randint(0, self.n_clusters, size=X.shape[0])

        # Finding clusters' centroids
        self.centroids_ = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])

        # Optimization
        for _ in range(self.max_iters):
            old_centroids = self.centroids_.copy()

            # Assign each data point to the nearest centroid
            self.labels_ = self._assign_labels(X)

            # Update centroids based on the mean of data points in each cluster
            centroids = []
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if len(cluster_points) > 0:
                    centroids.append(cluster_points.mean(axis=0))
                else:
                    # Handle empty clusters:
                    # Random centroid is assigned to empty clusters
                    centroids.append(np.random.rand(X.shape[1]))  # Random centroid

            self.centroids_ = np.array(centroids)

            # Check for convergence
            if np.allclose(self.centroids_, old_centroids):
                print("Converged")
                break
    
    def _assign_labels(self, X):
        """
        Assign labels to the data points based on the nearest centroid.

        Args:
            X (array-like): Input data with shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of labels for each data point.
        """
        if self.similarity == "cosine":
            distances = self._cosine_distances(X)
        elif self.similarity == "euclidean":
            distances = self._euclidean_distances(X)
        else:
            raise ValueError("Invalid similarity metric. Choose 'cosine' or 'euclidean'.")

        return np.argmin(distances, axis=1)

    def _cosine_distances(self, X):
        """
        Calculate cosine distances between data points and centroids.

        Args:
            X (array-like): Input data with shape (n_samples, n_features).

        Returns:
            np.ndarray: Cosine distances from each data point to each centroid.
        """
        dot_product = X @ self.centroids_.T
        norm_x = np.linalg.norm(X, axis=1, keepdims=True)
        norm_centroids = np.linalg.norm(self.centroids_, axis=1)
        cosine_similarity = dot_product / (norm_x * norm_centroids)
        return 1 - cosine_similarity

    def _euclidean_distances(self, X):
        """
        Calculate Euclidean distances between data points and centroids.

        Args:
            X (array-like): Input data with shape (n_samples, n_features).

        Returns:
            np.ndarray: Euclidean distances from each data point to each centroid.
        """
        return np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)

    def predict(self, samples):
        """
        Predict the closest cluster each sample in samples belongs to.

        Args:
            samples (array-like): Test data with shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted cluster for each sample.
        """
        
        return self._assign_labels(samples)
    

    def internal_cluster_dist(self):
        """
        Calculate the total internal cluster distance of the data points from the mean.

        Returns:
            float: Total internal cluster distance.
        """
        total_dist = 0
        for k in range(self.n_clusters):
            cluster_points = self.X[self.labels_ == k]
            if cluster_points.size > 0:
                dist = np.sum((cluster_points - self.centroids_[k]) ** 2)
                total_dist += dist

        return total_dist