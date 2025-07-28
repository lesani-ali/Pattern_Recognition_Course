import numpy as np
import matplotlib.pyplot as plt


class MMD:

    # Constructor
    def __init__(self):
        """
        Initialize the MMD (Maximum Mahalanobis Distance) classifier.
        """
        self.means = None
        self.inv_covariances = None

    def fit(self, X_train, y_train):
        """
        Fit the MMD classifier to the training data.

        Args:
            X_train (array-like): Training data, shape (n_samples, n_features).
            y_train (array-like): Target values, shape (n_samples,).
        """
        self.means = self._compute_means(X_train, y_train)
        self.inv_covariances = self._compute_inv_covariances(X_train, y_train)

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
                inv_covariance = self.inv_covariances[label]
                diff = x.reshape(-1, 1) - mean
                distances[label] = np.dot(np.dot(diff.T, inv_covariance), diff)
            predictions.append(min(distances, key=distances.get))
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

    def _compute_inv_covariances(self, X, y):
        """
        Compute the inverse of covariance matrices for each class.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).

        Returns:
            dict: Class labels as keys and inverse covariance matrices as values.
        """
        inv_covariances = {}
        for label in np.unique(y):
            covariance = np.cov(X[y == label], rowvar=False)
            inv_covariances[label] = np.linalg.inv(covariance)
        return inv_covariances

    def get_decision_boundary_parameters(self):
        """
        Calculates the parameters of decision boundary.

        Returns:
            tuple: Decision boundary parameters.
        """
        z1 = list(self.means.values())[0]  # Mean for class 1
        z2 = list(self.means.values())[1]  # Mean for class 2
        c1 = list(self.inv_covariances.values())[0]  # Cov inverse for class 1
        c2 = list(self.inv_covariances.values())[1]  # Cov inverse for class 2

        # Decison boundary: x.T A x + B x + C = 0
        A = c1 - c2
        B = 2 * (z2.T @ c2 - z1.T @ c1)
        C = z1.T @ c1 @ z1 - z2.T @ c2 @ z2

        return A, B, C

    # This method plots decision boundary for two dimmensional features
    def plot_decision_boundary_for2D(self, X_train, y_train):
        """
        Plots decision boundary for two dimensional features

        Args:
            X_train (array-like): Training data, shape (n_samples, n_features).
            y_train (array-like): Target values, shape (n_samples,).
        """

        # Decision boundary parameters
        A, B, C = self.get_decision_boundary_parameters()

        # Creating mesh for plot
        x1 = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
        x2 = np.linspace(np.min(X_train[:, 1]), np.max(X_train[:, 1]), 100)
        X1, X2 = np.meshgrid(x1, x2)

        g = lambda x: (x.T @ A @ x + B @ x + C).item()  # Decision boundary
        Z = np.array(
            [
                [g(np.array([x1, x2]).reshape(-1, 1)) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )

        # Extracting the data related to each class
        label = np.sort(np.unique(y_train))
        class1_points = X_train[y_train == label[0]]
        class2_points = X_train[y_train == label[1]]

        # Displaying the decision boundary
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(
            class2_points[:, 0],
            class2_points[:, 1],
            c="r",
            s=1,
            marker="x",
            edgecolors=None,
            label="Class " + str(label[1]),
        )
        ax.scatter(
            class1_points[:, 0],
            class1_points[:, 1],
            c="b",
            s=1,
            marker="x",
            edgecolors=None,
            label="Class " + str(label[0]),
        )
        ax.contour(X1, X2, Z, levels=[0], colors="k")
        ax.set_xlabel("x1", fontsize=12, fontname="sans-serif", labelpad=3)
        ax.set_ylabel("x2", fontsize=12, fontname="sans-serif", labelpad=3)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(frameon=False, loc=2, borderpad=0, labelspacing=0.15, fontsize=10)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.set_title("Decision Boundary for MMD")
        plt.show()
