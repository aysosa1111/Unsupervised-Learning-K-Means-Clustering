# Olivetti Faces Dataset Analysis

This project applies machine learning techniques to analyze the Olivetti faces dataset. It involves classification using a RandomForest classifier, dimensionality reduction with PCA, and clustering using K-Means and DBSCAN algorithms.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- scikit-learn

### Installation

1. Ensure Python 3.x is installed on your system. If not, download and install it from [Python's official site](https://www.python.org/downloads/).

2. Install the required Python libraries:

   ```bash
   pip install numpy scikit-learn
    ```

3. Clone the repository or download the source code.

   ```bash
   git clone https://github.com/aysosa1111/Unsupervised-Learning-K-Means-Clustering.git
    ```

4. Navigate to the project directory.

   ```bash
   cd path_to_your_project
    ```

## Running the Analysis
To run the analysis, execute the Python script from the command line:

   ```bash
  python K-Means_Clustering.py
  ```


# Project Structure
- K-Means_Clustering.py: Main script that contains all the code to perform the dataset loading, preprocessing, model training, and evaluation.

# Analysis Workflow
1. Data Loading: The Olivetti faces dataset is loaded from scikit-learn's dataset library.

2. Preprocessing: Data is split into training, validation, and test sets.

3. Model Training:

  - A RandomForest classifier is trained using k-fold cross-validation.
  - PCA is applied for dimensionality reduction.
  - K-Means clustering is performed to find optimal cluster numbers.
  - DBSCAN is used for additional clustering insights.
4. Evaluation:

- The classifier's performance is evaluated on the validation set.
- Clustering performance is assessed using the silhouette score.

5. Technical Notes: Address memory leak warnings in KMeans by setting the OMP_NUM_THREADS=1 environment variable, especially important for Windows users.

# Results
The RandomForest classifier achieves high accuracy on both the original and reduced datasets.
K-Means clustering identifies two main clusters but with a low silhouette score, indicating moderate separation.
DBSCAN mostly identifies outliers, suggesting parameter tuning is necessary.
