import numpy as np

class LinearRegression():
    def __init__(self, input_size):
        """
        Initialize weights and biases.
        
        Args:
            input_size (int): Number of input features.
        """
        self.W = np.random.randn(input_size, 1) * 0.01
        self.b = np.zeros((1, 1))

        self.dW = np.zeros((input_size, 1))
        self.db = np.zeros((1, 1))

    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        """
        Implement forward pass.
        
        Args:
            X (numpy.ndarray): Input data, batched along dimension zero.
        
        Returns:
            numpy.ndarray: Output of the forward pass (sigmoid of linear combination).
        """
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self.z
        return self.a

    def backward(self, X, y):
        """
        Implement backward pass, assuming the MSE loss.
        
        Args:
            X (numpy.ndarray): Input data, batched along dimension zero.
            y (numpy.ndarray): Batched target values.
        """
        m = X.shape[0]
        dz = self.a - y
        self.dW = np.dot(X.T, dz) / m
        self.db = np.sum(dz) / m

    def get_params_and_grads(self):
        """
        Return parameters and corresponding gradients.
        
        Returns:
            tuple: Parameters and gradients.
        """
        params = [self.W, self.b]
        grads = [self.dW, self.db]
        return params, grads
    
    def loss(self, y_true, y_pred):
        """
        Compute the mean square error (MSE) loss.
        
        Args:
            y_true (numpy.ndarray): Target values.
            y_pred (numpy.ndarray): Predicted values.
        
        Returns:
            float: MSE loss.
        """
        m = y_true.shape[0]
        loss = np.mean(0.5 * (y_pred - y_true)**2)
        return loss