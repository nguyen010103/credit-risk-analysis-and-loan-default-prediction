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

## Tools & Libraries 
- Python: pandas, numpy, sklearn
- SQL: PostSQL
- Visualization: matplotlib, seaborn

---

## Status 
- Project setup & environment

---

## Author
Nguyen Do
Aspiring Data Analyst 
www.linkedin.com/in/nguyenanhdo|https://github.com/nguyen010103|anhnguyen112003@gmail.com