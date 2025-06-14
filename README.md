Customer Churn Prediction Using Deep Learning

This project implements a deep learning-based binary classification model to predict customer churn using the Telco Customer Churn dataset. It also includes model explainability using LIME and SHAP to interpret the predictions.
Overview
Customer churn prediction helps businesses identify customers who are likely to stop using their services. This model leverages neural networks for prediction and explainable AI (XAI) tools to improve trust and transparency.
Dataset
Source: Telco Customer Churn dataset
Target variable: Churn Value (1 = Churned, 0 = Not Churned)
Features: Demographic, account, and service usage data
Preprocessing
Dropped irrelevant columns such as customer ID and location details
Converted 'Total Charges' to numeric and handled missing values
Encoded categorical features using LabelEncoder
Standardized numerical features using StandardScaler
Model Architecture
Input layer with shape based on number of features
Two dense layers with ReLU activation and dropout
Batch normalization for training stability
Output layer with sigmoid activation for binary classification
Compiled with Adam optimizer, binary crossentropy loss, and accuracy and AUC metrics
Training and Evaluation
Data split: 80% training, 20% testing
Epochs: 25
Batch size: 32
Evaluation metrics: Accuracy, AUC, Confusion Matrix, ROC Curve, and Classification Report
Explainability
LIME
Explains individual predictions by approximating the model locally with interpretable models
Visualized using matplotlib with color-coded bar charts showing feature contributions
SHAP
Provides global and local explanations using Shapley values
Highlights important features driving churn predictions
Visualization
Accuracy and loss curves over training epochs
ROC curve with AUC score
Enhanced LIME explanation plots for individual predictions
SHAP summary plots and force plots
