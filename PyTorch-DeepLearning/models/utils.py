import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader


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

# Function to calculate accuracy
def calculate_accuracy(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            # Send input to device
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model.predict(inputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)


def load_data(transform, train_val_ratio=0.8, batch_size=200):
    # Load MNIST dataset with the defined transformation
    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Define the sizes for the train and validation sets
    train_size = int(train_val_ratio * len(train_set))
    val_size = len(train_set) - train_size

    # Use random_split to create the train and validation sets
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    # Define dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    if train_val_ratio == 1:
        val_loader = test_loader
    else:
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return data_loader

# Function used for training the network
def train(num_epochs, data_loader, model, optimizer, criterion, device, verbose=False):
    train_loader = data_loader['train']
    val_loader = data_loader['val']

    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []

     # Training process
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set model to training mode
        for (inputs, labels) in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):

            # Send input to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate on validation data
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                # Send input to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Calculate training and validation accuracy
        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc = calculate_accuracy(model, val_loader, device)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}, 
                  'f'Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

    return train_losses, train_accuracy, val_losses, val_accuracy