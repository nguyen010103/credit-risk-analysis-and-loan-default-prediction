# Credit Risk Analysis & Loan Default Prediction

This project focuses on building a binary classification model to predict the likelihood of a borrower defaulting on a loan. It combines data cleaning, SQL exploration, machine learning, and model interpretability to create a robust, production-ready pipeline.

---

## Dataset

**Source:** [LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
**Target Variable:** `loan_status`  
**Goal:** Predict whether a loan will be *fully paid* or *charged off*.

---

## Objectives

- Explore and clean financial lending data
- Perform SQL-based data exploration and feature extraction
- Train ML models (Logistic Regression, XGBoost, etc.)
- Evaluate model performance using AUC, precision-recall, etc.
- Apply SHAP for model interpretability
- (Optional) Build a Streamlit dashboard for demo

---

## Exploratory Data

### Loan Status Distribution
- The majority of loans are either Fully Paid or Current, indicating successful repayment or active loans.
- Charged Off loans — used to define the target class — make up a minority of the data, confirming a significant class imbalance.
- Less frequent statuses (e.g., Late, Default, In Grace Period) were identified as rare and may require grouping or exclusion to simplify modeling.

### Target Variable - default
- Create a binary variable column
    - 1 = Charged Off (default)
    - 0 = all other loan issue

---

## Data Cleaning

### Filter Useful columns 
- Carefuly choosing important columns and import it into new dataframe
- Reduced from 152 columns to 25 meaningful columns

### Data Type Cleaning
- Remove % symbol from percentage feature such as int_rate, revol_util
- Parse emp_length as number of years
- Convert issue_d to datetime

### Missing Values
- Drop columns with more than 50% missing value
- Remove column with only on unique value
- Drop rows with missing key features

---

## Data Engineering and Preprocessing

### Engineered Features
- loan_to_income = loan_amnt / annual_inc
- credit_utilization = revol_bal * revol_util / 100
- debt_to_income = installment / (annual_inc / 12)
- log_annual_inc = log(annual_inc)

### Redundant Columns 
- Create visualization for data correlation
- Drop redundant columns such as 'fico_range_high', 'revol_bal', 'loan_to_income'

### Data Preprocessing
- One-hot encode nominal categorical variables
- Scaled numeric features using StandardScaler, excluding default column

### Outputs
- Clean saved engineered dataset under csv file

---

## Data Modeling 

### Linear Regression Model
- Trained using GridSearchCV model
- Shows moderate performance, with an ROC-AUC of 0.71
- Model captures 67% of actual defaulters (recall), its precision for the default class is low (0.31)

### Random Forest Classifier
- Trained as a benchmark model before XGBoost
- Performed reasonably well on accuracy and recall
- Helped establish baseline performance
- Served as a comparative reference during threshold tuning and final model selection

### Baseline XGBoost
- Trained an initial XGBoost classifier
- ROC-AUC score: **0.72**
- Initial F1 score: ~**0.39**
- Confusion Matrix:
  - True Positives: 33,312
  - False Negatives: 15,948
  - False Positives: 72,019
  - True Negatives: 132,078

### Threshold Tuning
- Evaluated model performance at different probability thresholds
- Identified optimal threshold: **0.62**
- Final scores:
  - **Precision:** 0.390
  - **Recall:** 0.428
  - **F1 Score:** 0.408
  - **Accuracy:** 76%

### Custom Evaluation Function
A reusable function was built inside script folder to:
- Plot confusion matrix
- Draw ROC and Precision-Recall curves
- Visualize F1, precision, and recall scores at different thresholds

### Key findings
- XGBoost improved recall and ROC-AUC over Random Forest.
- Threshold tuning helps adjust between false positives and false negatives.
- Cost-sensitive or business-specific threshold selection is crucial.

> This modular function enabled rapid evaluation of model improvements and guided the decision to adjust the classification threshold for better trade-offs between false positives and false negatives.

---

## Risk Scoring & Customer Segmentation

### Distribution of Risk Score
- Created new risk_score column in the dataframe using model predict probability
- Scaled risk_score column from 0 to 100
- Visualized the result using histogram
- Normal-shaped curve centered around 50
- Most borrowers fall into a moderate risk category (scores between 30–70).

### Risk Score by Loan Amount
- Grouped the loan_amount by risk_score and created the loan_bucket column
- Visualized the result using boxplot
- Median risk increase with loan size
- Borrowers requesting larger loans tend to carry higher risk.

### Risk Score by Income 
- Grouped the risk_score by loan_amount
- Visualized the result using bar chart
- Inverse relationship between income and risk.
- Lower-income borrowers are substantially riskier than high-income ones.

--- 

## Tools & Libraries 
- Python: pandas, numpy, sklearn
- Visualization: matplotlib, seaborn

---

## Status 
- Project setup & environment
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing
- Data Modeling

---

## Author
Nguyen Do
Aspiring Data Analyst 
www.linkedin.com/in/nguyenanhdo|https://github.com/nguyen010103|anhnguyen112003@gmail.com