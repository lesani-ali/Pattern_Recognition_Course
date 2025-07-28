import numpy as np


class GaussianMixture():
    def __init__(self, n_components, max_iters=500, tol=1e-5):
        """
        Initializes the Gaussian Mixture Model.

        Args:
            n_components (int): Number of mixture components.
            max_iters (int): Maximum number of iterations.
            tol (float): Convergence threshold.
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

    def fit(self, X):
        """
        Fits the Gaussian Mixture Model to the data.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).
        """
        n_features = X.shape[1]

        # Initialize means, covariances, and mixing coefficients
        pi = np.ones(self.n_components) / self.n_components # Shape: (n_components,)
        mu = np.random.rand(self.n_components, n_features) # Shape: (n_components, n_features)
        sigma = np.random.rand(self.n_components, n_features) # Shape: (n_components, n_features)

        # EM algorithm
        log_l_pr = 1 # log-likelihood for previous step
        for itr in range(self.max_iters):
            # E-step
            gamma = self._expectation(X, pi, mu, sigma)
            
            # Log-likelihood
            log_l = - np.sum(np.log(np.sum(gamma, axis=1)))
            if itr > 1 and (np.abs(log_l - log_l_pr) <= self.tol * np.abs(log_l)):
                # print("Converged")
                break

            # M-step
            pi, mu, sigma = self._maximization(X, gamma)

            log_l_pr = log_l

        self.means_ = mu
        self.covariances_ = sigma
        self.weights_ = pi

    
    def _expectation(self, X, pi, mu, sigma):
        """
        Performs the expectation step of the EM algorithm.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).
            pi (array-like): Weights of shape (n_components,).
            mu (array-like): Means of shape (n_components, n_features).
            sigma (array-like): Covariances of shape (n_components, n_features).

        Returns:
            gamma (array-like): Responsibilities of shape (n_samples, n_components).
        """
        n_samples = X.shape[0]

        # This is same as r we defined in class
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = pi[k] * self.multivariate_normal_pdf(X, mean=mu[k], cov=sigma[k])

        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return gamma

    def _maximization(self, X, gamma):
        """
        Performs the maximization step of the EM algorithm.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).
            gamma (array-like): Responsibilities of shape (n_samples, n_components).

        Returns:
            pi (array-like): Updated weights of shape (n_components,).
            mu (array-like): Updated means of shape (n_components, n_features).
            sigma (array-like): Updated covariances of shape (n_components, n_features).
        """
        n_samples, n_features = X.shape
        sum_gamma_k = np.sum(gamma, axis=0) # Shape: 1 by n_components

        # Update parameters
        pi = sum_gamma_k / n_samples # Shape: (n_components,)
        mu = (gamma.T @ X) / sum_gamma_k[:, np.newaxis] # Shape: n_components by n_features
        sigma = (gamma.T @ (X * X)) / sum_gamma_k[:, np.newaxis] - mu * mu # Shape: n_components by n_features

        return pi, mu, sigma
    

    def multivariate_normal_pdf(self, X, mean, cov):
        """
        Calculates the multivariate normal distribution.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).
            mean (array-like): Mean of the distribution of shape (n_features,).
            cov (array-like): Covariance matrix of shape (n_features,).

        Returns:
            pdf (array-like): Probability density values of shape (n_samples,).
        """
                
        n_features = X.shape[1]
        cov_inv = 1.0 / cov  # Inverse of diagonal elements of covariance matrix

        # Calculate the exponent term
        exponent = -0.5 * np.sum(((X - mean) ** 2) * cov_inv, axis=1)

        # Calculate the normalization term
        log_normalization = 0.5 * n_features * np.log(2 * np.pi) + 0.5 * np.sum(np.log(cov))
        normalization = np.exp(log_normalization)

        # Calculate the probability density
        pdf = np.exp(exponent) / normalization
        
        return pdf