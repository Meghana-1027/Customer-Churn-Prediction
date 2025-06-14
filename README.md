# Telco Customer Churn Prediction

## Overview

This project involves building a predictive model to identify customers likely to churn from a telecommunications service. Using the Telco Customer Churn dataset, we developed and trained a deep learning model using TensorFlow and Keras. Additionally, we incorporated model interpretability using LIME (Local Interpretable Model-agnostic Explanations) to better understand individual predictions.

## Dataset

The dataset used is the **Telco Customer Churn** dataset, which includes:
- Demographic data (gender, age, etc.)
- Services signed up for (internet, streaming, tech support, etc.)
- Account details (contract type, tenure, payment method, etc.)
- Billing information (monthly charges, total charges, etc.)
- Target variable: `Churn` (Yes/No)

## Preprocessing

- Removed irrelevant columns such as `CustomerID`, `Country`, `City`, etc.
- Handled missing values in `Total Charges` by converting to numeric and imputing median values.
- Encoded categorical features using `LabelEncoder`.
- Standardized numerical features using `StandardScaler`.
- Split data into training and testing sets with an 80:20 ratio.

## Model Architecture

The deep learning model includes:
- Input layer
- Two hidden layers with:
  - 128 and 64 neurons respectively
  - ReLU activation
  - Batch Normalization
  - Dropout layers (0.4 and 0.3 dropout rates)
- Output layer with sigmoid activation for binary classification

Compiled using:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy, AUC

## Evaluation & Results

- Model evaluated on the test set for Accuracy and AUC.
- Visualizations included:
  - Accuracy and loss plots over epochs
  - ROC Curve
  - Confusion Matrix
  - Classification Report

## Model Interpretability with LIME

To improve transparency, we used LIME for:
- Generating local explanations for individual predictions
- Visualizing feature impacts using custom matplotlib plots
- Helping stakeholders understand why a prediction was made for a given customer

## Dependencies

- Python 3.11+
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- LIME

You can install dependencies using:

```bash
pip install -r requirements.txt
