# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 02:39:55 2024

@author: Owner
"""

import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score

# Set the number of OMP threads to 1 to avoid MKL memory leak on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load the Olivetti faces dataset
def load_dataset():
    data = fetch_olivetti_faces()
    images = data.images
    labels = data.target
    return images, labels

# Split dataset into training, validation, and test sets
def split_data(images, labels):
    X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train a classifier using k-Fold cross validation
def train_classifier(X_train, y_train):
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X_train_flat, y_train, cv=5)
    clf.fit(X_train_flat, y_train)
    return clf, scores

# Perform PCA and find optimal number of clusters using silhouette score
def reduce_dimensionality_and_cluster(X_train):
    X_flat = X_train.reshape((X_train.shape[0], -1))
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_flat)
    return pca, X_reduced

# Apply DBSCAN for clustering
def apply_dbscan(X):
    dbscan = DBSCAN(eps=5, min_samples=5) 
    clusters = dbscan.fit_predict(X)
    return clusters

# Evaluate classifier on validation set
def evaluate_classifier(clf, X_val, y_val):
    X_val_flat = X_val.reshape((X_val.shape[0], -1))
    y_pred = clf.predict(X_val_flat)
    return accuracy_score(y_val, y_pred)

# Main execution
if __name__ == "__main__":
    images, labels = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)
    clf, scores = train_classifier(X_train, y_train)
    print("Cross-validation scores:", scores)

    val_accuracy = evaluate_classifier(clf, X_val, y_val)
    print("Validation Accuracy:", val_accuracy)
    
    pca, X_reduced = reduce_dimensionality_and_cluster(X_train)
    X_val_reduced = pca.transform(X_val.reshape((X_val.shape[0], -1)))  # Transform validation set

    best_score = 0
    best_k = 0
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best number of clusters: {best_k} with a silhouette score of {best_score}")

    # Retrain a classifier on the reduced dataset
    clf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_reduced.fit(X_reduced, y_train)
    reduced_val_accuracy = evaluate_classifier(clf_reduced, X_val_reduced, y_val)  # Now using the correctly transformed validation set
    print("Validation Accuracy on Reduced Data:", reduced_val_accuracy)
    
    clusters = apply_dbscan(X_reduced)
    print("Cluster labels from DBSCAN:", clusters)
