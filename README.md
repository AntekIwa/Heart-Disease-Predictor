# Heart Disease Prediction

This repository contains a machine learning project aimed at predicting the presence of heart disease based on patient data. Various classification models are trained and evaluated to identify the best-performing approach.

## Project Overview

The goal of this project is to predict whether a patient has heart disease using different machine learning classifiers. The project compares the performance of several models:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Data

The dataset used contains patient information with features relevant to heart disease risk. The data is split into training and test sets. Before modeling, features are scaled where necessary.

## Methodology

1. **Data Preparation**
   - Split dataset into `x_train`, `x_test`, `y_train`, `y_test`.
   - Feature scaling for models sensitive to feature magnitude (KNN, SVM).

2. **Model Training**
   - Train multiple classifiers with default or tuned hyperparameters.
   - Evaluate performance using metrics: Accuracy, Precision, Recall, F1-score, ROC AUC.

3. **KNN Analysis**
   - Evaluate KNN performance for a range of neighbors (`k=1` to `50`).
   - Select optimal `k` based on ROC AUC.

4. **Performance Evaluation**
   - Compare models using bar plots of metrics.
   - Visualize ROC curves for all models.
   - Examine confusion matrices to analyze misclassifications.

## Results

- Logistic Regression and KNN (k=50) showed the best balance between accuracy, recall, and ROC AUC.
- Ensemble methods (Random Forest, XGBoost) performed well, but slightly lower ROC AUC.
- High recall of KNN (k=50) makes it suitable for medical applications where identifying sick patients is critical.

## Visualizations

The notebook includes:

- **Bar plots** of Accuracy, F1-score, and ROC AUC for all models.
- **Line plot** showing the impact of `k` on KNN performance.
- **ROC curves** comparing models’ ability to distinguish between healthy and sick patients.
- **Confusion matrices** to inspect misclassifications.

