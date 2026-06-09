Loan Approval Prediction System
Overview

This project predicts whether a loan application will be Approved or Rejected using a Random Forest Classifier. The model is trained on a loan approval dataset containing applicant financial details and credit information.

Features
Data preprocessing using Pandas
Label Encoding for categorical variables
Train-Test Split for model evaluation
Random Forest Classification
Feature Importance Visualization
Loan Approval Prediction for new applicants
Dataset Attributes

The model uses the following features:

Number of Dependents
Annual Income
Loan Amount
Loan Term
CIBIL Score
Self Employed Status
Commercial Assets Value
Luxury Assets Value
Bank Asset Value

Target Variable:

Loan Status (Approved / Rejected)
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Installation

Install the required libraries:

pip install pandas numpy scikit-learn matplotlib
Project Workflow
Load the dataset.
Preprocess the data.
Encode categorical variables.
Split data into training and testing sets.
Train the Random Forest model.
Evaluate model accuracy.
Visualize feature importance.
Predict loan approval status for new applicants.
