# Machine Learning Assignment 2

## Problem Statement
Implement and compare exactly the five classification models specified in the assignment on the same public classification dataset.

## Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic?utm_source=chatgpt.com  
**Instances:** 569  
**Features:** 30  
**Task:** Binary classification  
**Positive class:** Malignant

## Repository Link
https://github.com/2025ac05197-wilp/ML-Assignment-2

## Models Implemented
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

## Evaluation Metrics
<img width="1026" height="221" alt="image" src="https://github.com/user-attachments/assets/9a7a9c37-4b25-42fe-a71d-7d66d142483a" />

## Observations on the performance of each model
<img width="1076" height="374" alt="image" src="https://github.com/user-attachments/assets/0362fd66-f590-4fc9-8faf-a57fb00a5d80" />

## Streamlit App Link
https://ml-assignment-2-models-classification.streamlit.app/

## Repository Structure
```text
ML-Assignment-2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
└── model/
    ├── common.py
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    └── random_forest.py
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
