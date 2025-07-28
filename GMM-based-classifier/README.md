# GMM-Based Classifier

This repository contains an implementation of a diagonal Gaussian Mixture Model (dGMM) based classifier where all covariance matrices are constrained to be diagonal, tested on the MNIST dataset. The project demonstrates the use of GMM for classification tasks and includes custom implementations of the GMM and the classifier.


## Project Structure
```
GMM-based-classifier/
├── models/                         # Directory containing class implementations
│   ├── __init__.py     
│   ├── GaussianMixture.py          # diagonal Gaussian Mixture Model (dGMM) implementation
│   ├── GMMBasedClassifier.py       # GMM-based classifier implementation
│   ├── utils.py                    # Utility functions implementation
├── main.ipynb                      # Notebook demonstrating the use of classifier
├── .gitignore                      # Git ignore file
├── environment.yml                 # Python dependencies
└── README.md                       # Project README file
```


## Acknowledgements
- The MNIST dataset is provided by Yann LeCun and can be found [here](http://yann.lecun.com/exdb/mnist/).
