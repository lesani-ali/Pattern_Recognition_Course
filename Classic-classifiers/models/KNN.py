import numpy as np


class KNN:

    # Constructor
    def __init__(self, k=5, _type="classification"):
        """
        Initialize k-Nearest Neighbors (KNN) with specified parameters.

        Args:
            k (int): Number of neighbors to use.
            _type (str): Either 'classification' or 'regression'.
        """
        self.k = k
        self._type = _type  # 'classification' or 'regression'

    def fit(self, X_train, y_train):
        """
        Fit the KNN model to the training data.

        Args:
            X_train (array-like): Training data, shape (n_samples, n_features).
            y_train (array-like): Target values, shape (n_samples,).
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict the class labels for the input samples.

        Args:
            X_test (array-like): Test data, shape (n_samples, n_features).

        Returns:
            numpy array: Predicted labels.
        """
        y_predicted = np.array([self._predict(sample.reshape(-1, 1)) for sample in X_test])
        return y_predicted

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Args:
            x (array-like): Input sample, shape (1, n_features).

        Returns:
            predicted label
        """
        distances = [np.linalg.norm(x - sample.reshape(-1, 1)) for sample in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]

        if self._type == "classification":
            prediction = max(set(k_nearest_labels.tolist()), key=k_nearest_labels.tolist().count)
        elif self._type == "regression":
            prediction = np.mean(k_nearest_labels)
        else:
            raise ValueError(
                "Invalid type specified. Use 'classification' or 'regression'."
            )

        return prediction
