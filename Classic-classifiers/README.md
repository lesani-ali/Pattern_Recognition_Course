# MNIST Classification with MED, MMD, KNN, ML, and MAP

This repository contains implementations of several classifiers, including Minimum Euclidean Distance (MED), Maximum Mahalanobis Distance (MMD), k-Nearest Neighbors (KNN), Maximum Likelihood (ML), and Maximum A Posteriori (MAP). These classifiers are applied to a subset of the MNIST dataset. The dataset has been reduced to include only two classes (digits 3 and 4), and dimensionality reduction to 2 dimensions has been performed using PCA. Additionally, a regression version of KNN is implemented and applied to the dataset stored in the "regression_data" folder.

## Project Structure
```
Classic-classifiers/
├── regression_data/           # Directory for datasets for regression
├── models/                    # Directory containing class implementations
│   ├── __init__.py
│   ├── MED.py                 # Minimum Euclidean Distance (MED) classifier implementation
│   ├── MMD.py                 # Maximum Mahalanobis Distance (MMD) classifier implementation
│   ├── KNN.py                 # k-Nearest Neighbors (KNN) classifier/regression implementation
│   ├── MAP.py                 # Maximum A Posteriori (MAP) classifier implementation
│   ├── ML.py                  # Maximum Likelihood (ML) classifier implementation
├── main.ipynb                 # Notebook demonstrating the use of classifiers
├── knn_regression.ipynb       # Notebook demonstrating the KNN regression
├── .gitignore                 # Git ignore file
├── environment.yml            # Python dependencies
└── README.md                  # Project README file
```


## Acknowledgements
- The MNIST dataset is provided by Yann LeCun and can be found [here](http://yann.lecun.com/exdb/mnist/).
