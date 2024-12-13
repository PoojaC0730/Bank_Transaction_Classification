# Bank_Transaction_Classification
Bank_Transaction_Classification

#Project Overview
This project involves classifying bank transactions into predefined ledger tags using machine learning. The key tasks include data preprocessing, feature engineering, model training, evaluation, and prediction for unseen transactions. The best model is saved and can be reused for classifying future transactions.

##How to Run the Code
•  Dependencies:
•	Python 3.8+
•	pandas
•	scikit-learn
•	joblib
•	matplotlib
•	seaborn

Install dependencies using:
1.	Steps to Run:
o	Data Preparation and Cleaning : Run the Data_Cleaning.ipynb file with “bank.xlsx” as the training/testing data. This will save the cleaned data into a csv file. 
o	Feature Engineering and Model Training and Saving: Run the “Feature Engineering and Model Fitting and Evaluations.ipynb” script to train models on the provided clean and tagged dataset. As the dataset is manually tagged, use the “tagged_cleandata.csv” file and save the best-performing model.
o	Prediction on New Data: Use the saved model to classify new transactions by running the Predict_LedgerTags.ipynb file on the unclean and raw data “bank_lastquarter.xlsx”

##Project Workflow
1.	Data Preprocessing:
o	Cleaning missing values.
o	Removing irrelevant columns.
2.	Feature Engineering:
o	TF-IDF vectorization on transaction details.
3.	Model Training:
o	Training multiple models (Random Forest, Logistic Regression, K-Nearest Neighbors, and SVM).
o	Selecting the best model based on weighted F1 score.
4.	Error Analysis:
o	Generating classification reports and confusion matrices.
5.	Model Saving:
o	Saving the best model, TF-IDF vectorizer, and label encoder for future use.
6.	Prediction:
o	Classifying new transactions using the saved model.

##Data Documentation
###Data Preprocessing
•	Removed unnecessary columns that were not affecting the ledger tags (category) of a given transaction. 
•	Cleaned up rows with null values for transaction details. 
•	Filled null values in Amount related columns with 0. 
•	Converted date column into proper format. 

###Data Transformation
•	Used TF-IDF Vectorizer to convert TRANSACTION DETAILS into numerical features.
o	Parameters:
	max_features=500
	ngram_range=(1, 2)
	stop_words='english'
•	Encoded target labels (Category) using scikit-learn's LabelEncoder.

##Model Documentation

###Model Selection Logic
We trained the following models to classify transactions:
1.	Random Forest:
o	Strengths: Handles high-dimensional data well, robust to overfitting.
o	Applicability: Useful for handling datasets with mixed feature importance and non-linear relationships, such as transaction data with diverse patterns.
o	Hyperparameters:
	n_estimators=100
	random_state=42
2.	Logistic Regression:
o	Strengths: Performs well with linearly separable data.
o	Applicability: Included as a baseline model to observe performance with simpler, linear assumptions about the data.
o	Hyperparameters:
	max_iter=1000
	random_state=42
3.	K-Nearest Neighbors (KNN):
o	Strengths: Simple and interpretable.
o	Applicability: Effective for small datasets with locally concentrated patterns, although performance can degrade with high-dimensional feature spaces.
o	Hyperparameters:
	Default (k=5).
4.	Support Vector Machine (SVM):
o	Strengths: Effective in high-dimensional spaces, works well with clear margins.
o	Applicability: Chosen to evaluate performance in identifying distinct boundaries between transaction tags with limited overlap.
o	Hyperparameters:
	Kernel: Linear
	random_state=42

###Model Evaluation
•	Metrics Used:
o	Accuracy
o	Precision, Recall, and F1 Score
•	Best Model:
o	Selected based on the highest F1 score.
o	Saved the model pkl file using joblib.

###Error Analysis
•	Techniques:
o	Generated confusion matrices to identify misclassified categories.
o	Analyzed incorrect predictions to identify patterns and improve features.
•	Observations:
o	Tags with insufficient training data had lower performance.
o	Overlapping or ambiguous transaction details caused misclassifications.
