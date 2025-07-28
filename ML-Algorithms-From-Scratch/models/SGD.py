class SGD():
    def __init__(self, params, learning_rate):
        """
        Initialize the SGD optimizer.
        
        Args:
            params (list): List of parameters to optimize.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.params = params
        self.lr = learning_rate

    def step(self, grads):
        """
        Perform one step of SGD.
        
        Args:
            grads (list): List of gradients for the parameters.
        """
        for param, grad in zip(self.params, grads):
            param -= self.lr * grad