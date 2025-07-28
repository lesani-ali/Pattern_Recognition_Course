import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from classes.LinearRegression import LinearRegression

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

def plot_all_results(all_losses, all_accuracies, all_labels):
    if len(all_losses) != len(all_accuracies):
        raise ValueError("all_losses length must be equal to all_accuracies length")

    if len(all_losses) != len(all_labels):
        raise ValueError("all_labels length must be equal to all_losses length")

    epochs = len(all_losses[0])
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(len(all_losses)):
        plt.plot(range(1, epochs + 1), all_losses[i], label=all_labels[i])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training loss')

    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(len(all_losses)):
        plt.plot(range(1, epochs + 1), all_accuracies[i], label=all_labels[i])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Training accuracy')

    plt.legend()
    plt.show()

def gen_batches(data, labels, batch_size):
    """
    Generate batches from the dataset.
    
    Args:
        data (numpy.ndarray): Features.
        labels (numpy.ndarray): Labels.
        batch_size (int): Size of each batch.
        
    Yields:
        tuple: Batches of (X, y).
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size], labels[i:i+batch_size]

    
def compute_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    
    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
    
    Returns:
        float: Accuracy score.
    """
    return np.mean(y_true == y_pred)


def load_mnist():
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])  # Converting image to tensor
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)  # Loading train data
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)  # Loading test data

    # Preprocess the data
    X_train = train_dataset.data.view(train_dataset.data.size(0), -1).numpy()  # Flattening the images
    y_train = train_dataset.targets.numpy()  # Converting tensor to numpy array

    X_test = test_dataset.data.view(test_dataset.data.size(0), -1).numpy()  # Flattening the images
    y_test = test_dataset.targets.numpy()  # Converting tensor to numpy array


    # Masking data to extract only class 3 and 4
    class1 = 3
    class2 = 4
    training_mask = (y_train == class1) | (y_train == class2)  # Mask training data
    X_train = X_train[training_mask, :]  # Input data (images)
    y_train = y_train[training_mask]  # Labels

    test_mask = (y_test == class1) | (y_test == class2)  # Mask test data
    X_test = X_test[test_mask, :]  # Input data (images)
    y_test = y_test[test_mask]  # Labels


    # Applying PCA
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Normalizing data to be between 0 and 1
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    # Re-encoding labels
    y_train[y_train == 3] = 1
    y_train[y_train == 4] = 0

    y_test[y_test == 3] = 1
    y_test[y_test == 4] = 0

    # Splitting the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    data = {}
    data["train"] = (X_train, y_train)
    data["val"] = (X_val, y_val)
    data["test"] = (X_test, y_test)

    return data

def load_synthetic_regression_data(num_samples, a, b):
    # y = ax + b
    X_train = 2 * np.random.rand(num_samples, 1)
    y_train = a * X_train + b + np.random.randn(num_samples, 1)

    X_test = 2 * np.random.rand(50, 1)
    y_test = a * X_test + b + np.random.randn(50, 1)

    # Splitting the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    data = {}
    data["train"] = (X_train, y_train)
    data["val"] = (X_val, y_val)
    data["test"] = (X_test, y_test)

    return data

# Training loop
def train(model, data, optimizer, epochs, batch_size):
    X_train, X_val, y_train, y_val = data
    train_losses = []
    val_losses = []

    is_classification_model = not isinstance(model, LinearRegression)
    if is_classification_model:
        val_accuracies = []
        train_accuracies = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        random_indices = np.random.permutation(list(range(X_train.shape[0])))
        X_train = X_train[random_indices]
        y_train = y_train[random_indices]
        for X, y in gen_batches(X_train, y_train, batch_size):
            # Forward pass
            output = model(X)

            # Backward pass
            model.backward(X, y)

            # Get parameters and gradients
            params, grads = model.get_params_and_grads()

            # Update parameters using the chosen optimizer
            optimizer.step(grads)

        # Compute training loss and accuracy
        output = model(X_train)
        y_predict_train = (output >= 0.5).astype(int)
        train_loss = model.loss(y_train, output)

        # Compute test loss and accuracy
        output = model(X_val)
        y_predict_val = (output >= 0.5).astype(int)
        val_loss = model.loss(y_val, output)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if is_classification_model:
            train_accuracy = compute_accuracy(y_train, y_predict_train)
            val_accuracy = compute_accuracy(y_val, y_predict_val)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

    if is_classification_model:
        return train_losses, train_accuracies, val_losses, val_accuracies
        
    return train_losses, val_losses