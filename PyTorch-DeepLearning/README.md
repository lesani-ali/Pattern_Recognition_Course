# Deep Learning with PyTorch

This repository provides implementations of Convolutional Neural Networks (VGG11) and Multi-Layer Perceptrons (MLPs) using PyTorch. The models are trained and evaluated on the MNIST dataset, with results visualized and documented in the `main.ipynb` Jupyter Notebook. 


## Implemented Algorithms

1. VGG11 (CNN)
2. Multi-Layer Perceptron (MLP)


## Project Structure
```
PyTorch-DeepLearning/
├── models/                      # Directory containing class implementations
│   ├── __init__.py
│   ├── CNN.py                   # Implementation VGG11
│   ├── MLP.py                   # Implementation of Multi-Layer Perceptron
│   ├── utils.py                 # Utility functions (load_data, calculate_accuracy, load_data, train, etc)
├── main.ipynb                   # Notebook demonstrating the use of algorithms
├── .gitignore                   # Git ignore file
├── environment.yml              # Python dependencies
└── README.md                    # Project README file
```

## Acknowledgements
- The MNIST dataset is provided by Yann LeCun and can be found [here](http://yann.lecun.com/exdb/mnist/).
- The VGG11 architecture can be found [here](https://arxiv.org/pdf/1409.1556.pdf).
