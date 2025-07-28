import numpy as np
from .GaussianMixture import GaussianMixture


class GMMBasedClassifier():
    def __init__(self, n_components, max_iters=500, tol=1e-5):
        """
        Initializes the GMM-based classifier.

        Args:
            n_components (int): Number of mixture components.
            max_iters (int): Maximum number of iterations for GMM.
            tol (float): Convergence threshold for GMM.
        """

        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.models = {} # Data structure to store mixture model for each class
        self.cls_prior = {} # This is P(Y = c) which is prior probability of each class

    def fit(self, X_train, y_train):
        """
        Fits the GMM-based classifier to the data.

        Args:
            X_train (array-like): Training data of shape (n_samples, n_features).
            y_train (array-like): Target values of shape (n_samples,).
        """
        self.classes_ = np.unique(y_train)

        for clss in self.classes_:
            X_clss = X_train[y_train == clss]

            gmm = GaussianMixture(n_components=self.n_components)
            gmm.fit(X_clss)

            self.models[clss] = gmm
            self.cls_prior[clss] = X_clss.shape[0] / X_train.shape[0]

    def predict(self, X_test):
        """
        Predicts the class labels for the test data.

        Args:
            X_test (array-like): Test data of shape (n_samples, n_features).

        Returns:
            array: Predicted class labels of shape (n_samples,).
        """
        n_samples = X_test.shape[0] 
        n_classes = len(self.classes_)
        
        likelihoods = np.zeros((n_samples, n_classes))
        for i, clss in enumerate(self.classes_):
            gmm = self.models[clss]
            pi = gmm.weights_
            mu = gmm.means_
            sigma = gmm.covariances_

            p = 0
            for k in range(self.n_components):
                p += pi[k] * gmm.multivariate_normal_pdf(X_test, mean=mu[k], cov=sigma[k])                

            likelihoods[:, i] = self.cls_prior[clss] * p
        
        max_idx = np.argmax(likelihoods, axis=1)

        return self.classes_[max_idx]