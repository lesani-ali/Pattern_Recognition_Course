import numpy as np


class ML:

    # Constructor
    def __init__(self):
        """
        Initialize ML (Maximum Likelihood) classifier.
        """
        self.means = None
        self.covariances = None
        self.ksi = 1

    def fit(self, X_train, y_train):
        """
        Fit the ML classifier to the training data.

        Args:
            X_train (array-like): Training data, shape (n_samples, n_features).
            y_train (array-like): Target values, shape (n_samples,).
        """
        self.means = self._compute_means(X_train, y_train)
        self.covariances = self._compute_covariances(X_train, y_train)

    def predict(self, X_test):
        """
        Predict the class labels for the test data.

        Args:
            X_test (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array: Predicted class labels, shape (n_samples,).
        """
        predictions = []
        for x in X_test:
            distances = {}
            for label in self.means:
                mean = self.means[label]
                covariance = self.covariances[label]
                inv_covariance = np.linalg.inv(covariance)
                diff = x.reshape(-1, 1) - mean
                distances[label] = np.dot(np.dot(diff.T, inv_covariance), diff)

            label1, cov1 = list(self.covariances.items())[0]
            label2, cov2 = list(self.covariances.items())[1]

            determinant_ratio = np.linalg.det(cov1) ** 0.5 / np.linalg.det(cov2) ** 0.5
            eta = np.log(self.ksi * determinant_ratio)

            score_diff = -0.5 * distances[label1] + 0.5 * distances[label2]
            predictions.append(label1 if score_diff >= eta else label2)

        return np.array(predictions)

    def _compute_means(self, X, y):
        """
        Compute the means for each class.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).

        Returns:
            dict: Class labels as keys and means as values.
        """
        means = {}
        for label in np.unique(y):
            means[label] = X[y == label].mean(axis=0).reshape(-1, 1)
        return means

    def _compute_covariances(self, X, y):
        """
        Compute the covariances for each class.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).

        Returns:
            dict: Class labels as keys and covariances as values.
        """
        covariances = {}
        for label in np.unique(y):
            covariances[label] = np.cov(X[y == label], rowvar=False)

        return covariances
