import numpy as np

def normalize_data(data):
    """
    Normalizes the data.

    Args:
        data (array-like): The data to be normalized.

    Returns:
        array-like: The normalized data.
    """
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)

    range_values = max_ - min_
    range_values[range_values == 0] = 1 

    normalized_data = (data - min_) / range_values

    return normalized_data