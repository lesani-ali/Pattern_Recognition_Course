import numpy as np

class LogisticRegression():
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
    
    def sigmoid(self, z):
        """
        Apply the sigmoid function.
        
        Args:
            z (numpy.ndarray): Linear combination of inputs and weights.
        
        Returns:
            numpy.ndarray: Sigmoid of input.
        """
        return 1 / (1 + np.exp(-z))
    
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
        self.a = self.sigmoid(self.z)
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
        Compute the binary cross-entropy loss.
        
        Args:
            y_true (numpy.ndarray): True labels.
            y_pred (numpy.ndarray): Predicted probabilities.
        
        Returns:
            float: Binary cross-entropy loss.
        """
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss