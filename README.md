# AIML-Assignment
📊 Telco Customer Churn Prediction – AIML Project

🔍 # Project Overview
This project builds a machine learning model to predict customer churn (whether a customer will leave the service or not) using the Telco Customer Churn dataset.
The goal is to demonstrate basic ML workflow, data preprocessing, model training, and evaluation.

📁 Dataset Source:-
Dataset Name: Telco Customer Churn
Link:- https://www.kaggle.com/datasets/blastchar/telco-customer-churn
File Format: Excel (.xlsx)
Source: IBM Sample Dataset (commonly used for churn analysis)
Description:
The dataset includes customer details such as:
Gender, senior citizen status, partner and dependents
Tenure and subscribed services (Phone, Internet, Streaming, etc.)
Contract type, payment method, monthly and total charges
Target variable: Churn (Yes / No)

🛠️ Technologies & Libraries Used:-
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn


📑 Steps Performed in the Project:-
1) Imported essential Python libraries for:
   Data manipulation (pandas, numpy)
   Data visualization (matplotlib, seaborn)
   Machine learning (scikit-learn)

2) Load the Dataset:-
   Loaded the Telco Customer Churn CSV file using Pandas.
   Added error handling to ensure the dataset is available before execution.

3) Initial Data Exploration:-
   Displayed the first few rows of the dataset.
   Checked:
   Dataset shape (rows & columns)
   Column names
   Data types
   Missing values
   Performed basic understanding of the dataset structure.

4) Data Cleaning:-
   Converted TotalCharges column to numeric format.
   Handled missing values by removing invalid rows.
   Ensured all numerical columns were correctly formatted.

5) Categorical Data Encoding:-
   Converted categorical variables into numerical format using:
   Label Encoding
   Converted the target column Churn into binary values:
   Yes → 1
   No → 0

6) Feature Selection:-
   Removed unnecessary columns such as:
   customerID
   Selected relevant features for training the model.

7) Train-Test Split:-
   Split the dataset into:
   Training set
   Testing set
   Used 80% training and 20% testing data.
       
8) Feature Scaling:-
   Applied StandardScaler to normalize numerical features.
   Scaling ensured better model performance and convergence.

9) Model Building:-
   Implemented three machine learning classification algorithms to predict customer churn:
   Logistic Regression
   Decision Tree Classifier
   Random Forest Classifier
   Each model was trained using the training dataset after preprocessing and feature scaling (where required).

10) Model Prediction:-
    Generated predictions on the test dataset for all three models.
    Converted predicted probabilities into binary class labels (Churn / No Churn).     

11) Model Evaluation:-
    Evaluated each model using the following metrics:
    Accuracy Score
    Confusion Matrix
    Additionally:
    Visualized confusion matrices using heatmaps for better interpretation.
    Compared model performances based on evaluation results.

12) Result Interpretation:-
    For each model, the following were analyzed:
    True Positives (TP) – Correctly predicted churned customers
    True Negatives (TN) – Correctly predicted non-churned customers
    False Positives (FP) – Non-churned customers predicted as churn
    False Negatives (FN) – Churned customers predicted as non-churn
    This analysis helped understand strengths and weaknesses of each model in churn prediction.    
   
  Final Results

  All three models were successfully trained and evaluated.
  Random Forest achieved the best performance among the models.
  Logistic Regression provided a strong baseline with good interpretability.
  Decision Tree offered clear decision rules but was more prone to overfitting.
  The confusion matrices demonstrate each model’s ability to distinguish between churned and non-churned customers.   

  Conclusion
  This project demonstrates a complete end-to-end machine learning workflow, including:
  Data preprocessing
  Feature encoding and scaling
  Training multiple models
  Model evaluation and comparison



  
