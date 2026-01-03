# AIML-Assignment

## 📊 Telco Customer Churn Prediction – AIML Project

---

## 🔍 Project Overview
This project builds a machine learning model to predict customer churn (whether a customer will leave the service or not) using the Telco Customer Churn dataset.  
The goal is to demonstrate a basic ML workflow, data preprocessing, model training, and evaluation.

---

## 📁 Dataset Source
- **Dataset Name:** Telco Customer Churn
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **File Format:** Excel (.xlsx)
- **Source:** IBM Sample Dataset (commonly used for churn analysis)

### Description
The dataset includes customer details such as:
- Gender, senior citizen status, partner and dependents
- Tenure and subscribed services (Phone, Internet, Streaming, etc.)
- Contract type, payment method, monthly and total charges
- **Target variable:** Churn (Yes / No)

---

## 🛠️ Technologies & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📑 Steps Performed in the Project

### 1. Import Required Libraries
- Data manipulation: pandas, numpy
- Data visualization: matplotlib, seaborn
- Machine learning: scikit-learn

---

### 2. Load the Dataset
- Loaded the Telco Customer Churn CSV file using Pandas
- Added error handling to ensure the dataset is available before execution

---

### 3. Initial Data Exploration
- Displayed the first few rows of the dataset
- Checked:
  - Dataset shape (rows & columns)
  - Column names
  - Data types
  - Missing values
- Performed basic understanding of the dataset structure

---

### 4. Data Cleaning
- Converted `TotalCharges` column to numeric format
- Handled missing values by removing invalid rows
- Ensured all numerical columns were correctly formatted

---

### 5. Categorical Data Encoding
- Converted categorical variables into numerical format using Label Encoding
- Converted the target column `Churn` into binary values:
  - Yes → 1
  - No → 0

---

### 6. Feature Selection
- Removed unnecessary columns such as:
  - customerID
- Selected relevant features for training the model

---

### 7. Train-Test Split
- Split the dataset into training and testing sets
- Used 80% training and 20% testing data

---

### 8. Feature Scaling
- Applied StandardScaler to normalize numerical features
- Scaling ensured better model performance and convergence

---

### 9. Model Building
- Implemented the following classification models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Trained all models on the training dataset

---

### 10. Model Prediction
- Generated predictions on the test dataset for all three models
- Converted predicted probabilities into binary class labels

---

### 11. Model Evaluation
- Evaluated each model using:
  - Accuracy Score
  - Confusion Matrix
- Visualized confusion matrices using heatmaps
- Compared model performances

---

## ✅ Final Results
- All three models were successfully trained and evaluated
- Random Forest achieved the best performance
- Logistic Regression provided a strong baseline
- Decision Tree was interpretable but prone to overfitting

---

## 🏁 Conclusion
This project demonstrates a complete end-to-end machine learning workflow including:
- Data preprocessing
- Feature encoding and scaling
- Training multiple models
- Model evaluation and comparison
