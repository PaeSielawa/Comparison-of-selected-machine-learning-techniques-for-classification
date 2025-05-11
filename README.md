# Yogurt Analysis Project

## Overview
This project focuses on analyzing yogurt data using various machine learning techniques including dimensionality reduction and supervised classification methods. The analysis aims to identify patterns and relationships in yogurt characteristics across different parameters.

## Table of Contents
- Project Structure
- Features
- Requirements
- Installation
- Usage
- Analysis Methods
- Data Processing
- Results Visualization
- License

## Project Structure 

```
Comparison-of-selected-machine-learning-techniques-for-classification/
│
├── data/
│   ├── jogurt_kefir_synchr.csv    # First yogurt dataset
│   ├── jogurt_kefir_synchr1.csv   # Second yogurt dataset
│   └── jogurt_kefir_synchr2.csv   # Third yogurt dataset
│
├── notebooks/
│   ├── Projekt.ipynb              # Main analysis notebook
│   ├── ProjektFullData.ipynb      # Full dataset analysis
│   └── ProjektOneData.ipynb       # Single dataset analysis
│
├── src/
│   └── funkcje.py                 # Core functions and utilities
│
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
```

## Features
**Data Analysis**
- Data preprocessing and cleaning
- Feature extraction
- Pattern recognition
- Statistical analysis
- Time series analysis

**Dimensionality Reduction Methods**
- Principal Component Analysis (PCA)
- Kernel PCA
- Non-metric Multidimensional Scaling (NMDS)
- Uniform Manifold Approximation and Projection (UMAP)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Independent Component Analysis (ICA)
- Linear Discriminant Analysis (LDA)
- Multiple Discriminant Analysis (MDA)

**Classification Methods**
- k-Nearest Neighbors (kNN)
- Random Forest
- Support Vector Machines (SVM)
- Naive Bayes
- Logistic Regression

**Visualization Techniques**
- Scatter plots
- Confidence ellipses
- Convex hulls
- Dendrograms
- Confusion matrices

## Requirements
- Python 3.8+
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- umap-learn
- scipy

## Installation
1. Clone the repository:
```sh
git clone https://github.com/Comparison-of-selected-machine-learning-techniques-for-classification/Projekt.git
cd Projekt
```
2. Install packages individually:
```sh
pip install pandas numpy seaborn matplotlib scikit-learn umap-learn scipy
```

## Usage
**Data Processing**
```python
from funkcje import rename_unnamed_columns, replace_name, split_column_names

# Load data
df = pd.read_csv('jogurt_kefir_synchr.csv', encoding="ISO-8859-1", low_memory=False)

# Process column names
df = rename_unnamed_columns(df, start=1)
```

**Running Analysis**
```python
# Dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

## Analysis Methods
**Dimensionality Reduction**
- **PCA**: Linear dimensionality reduction
- **Kernel PCA**: Non-linear dimensionality reduction
- **NMDS**: Non-metric multidimensional scaling
- **UMAP**: Non-linear dimensionality reduction preserving local structure
- **t-SNE**: Non-linear dimensionality reduction for visualization
- **ICA**: Source separation and feature extraction
- **LDA**: Supervised dimensionality reduction
- **MDA**: Multiple discriminant analysis for multi-class problems

**Classification Methods**
- **kNN**: Non-parametric classification
- **Random Forest**: Ensemble learning method
- **SVM**: Support vector classification
- **Naive Bayes**: Probabilistic classification
- **Logistic Regression**: Linear classification

## Data Processing
1. Data Loading:
    - Multiple CSV files processing
    - Encoding handling
    - Missing data management
2. Data Cleaning:
    - Column renaming
    - Data type conversion
    - Missing value handling
3. Feature Engineering:
    - Time series feature extraction
    - Statistical feature computation
    - Pattern recognition

## Results Visualization
- The project includes various visualization methods:
- Interactive plots using matplotlib and seaborn
- Confidence level ellipses for uncertainty visualization
- Convex hull plotting for cluster boundaries
- Hierarchical clustering dendrograms
- Classification performance metrics visualization

## License

This project was created by PS. All rights reserved.

