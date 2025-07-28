from ML import ML

class MAP(ML):

    # Constructor
    def __init__(self, pc1, pc2):
        """
        Initialize MAP (Maximum A Posteriori) classifier with prior probabilities.

        Args:
            pc1 (float): Prior probability of class 1.
            pc2 (float): Prior probability of class 2.
        """
        super().__init__()
        self.ksi = pc2 / pc1
