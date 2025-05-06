# Mitigating Bias in Job Recommendation Systems

A fairness-aware Machine Learning job recommendation platform with a React.js frontend and Python backend, designed to detect and reduce algorithmic bias in employment suggestions.

![Alt Text](https://raw.githubusercontent.com/riddhi-testcases/Job-Recommendation-System/main/public/1..jpg)
![Alt Text](https://raw.githubusercontent.com/riddhi-testcases/Job-Recommendation-System/main/public/2..jpg)
![Alt Text](https://raw.githubusercontent.com/riddhi-testcases/Job-Recommendation-System/main/public/3..jpg)

---

## 🚀 Project Overview

Traditional ML-based job recommendation systems can exhibit bias against demographic groups. This project aims to build a fair, interpretable, and user-friendly job recommender that ensures equitable opportunities for all users.

---

## 🎯 Objectives

- Develop a job recommendation model that prioritizes fairness across gender and race.
- Apply counterfactual analysis to uncover hidden biases.
- Visualize fairness metrics and model predictions interactively via a React frontend and integrate it to the backend.

---

## 🧰 Tech Stack

- **Frontend:** React.js, Chart.js
- **Backend:** Python, REST API, Joblib
- **Machine Learning:** scikit-learn, Fairlearn, Pandas, NumPy
- **Visualization & Charts:** Matplotlib, Chart.js
- **Deployment:** Local (with options for Heroku/Render)

---

## 📊 Dataset

- **Source:** Demographic Employment Dataset (~2GB)
- **Features:** Age, Gender, Race, Education, Job Role, Experience, Salary
- **Preprocessing:**
  - Handled missing values (mean/mode)
  - Removed irrelevant features (ID, Address)
  - Encoded categorical fields (One-hot, Label Encoding)

---

## ⚙️ Counterfactual Analysis

To detect bias:
- Duplicated entries and swapped sensitive features (e.g., Male ↔ Female)
- Observed outcome changes to determine fairness
- Ensured minimal prediction disparity when only protected attributes were changed 

---

## 🤖 Model

- **Final Model:** Random Forest Classifier  
- **Why?** High accuracy (~88%), interpretable, less prone to overfitting  
- **Validation:** 5-fold CV, Accuracy, F1, ROC AUC

---


## 📈 Model Training & Evaluation

- **Split:** 80% Training / 20% Testing
- **Cross-validation:** 5-fold
- **Performance Metrics:** Accuracy (~88%), F1-Score, ROC AUC
- **Fairness Metrics:**
  - Disparate Impact (~0.91)
  - Equalized Odds
  - KL Divergence
  - Diversity Score

## 🌐 API & Frontend

- Backend: Model training, Fairness logic, REST API for predictions and fairness reports  
- Frontend: React UI, Charts, Upload CSV, view predictions & fairness charts  
- Secure and scalable structure; local deployment with cloud-ready setup

---

## 📊 Key Features

- Bias detection through interactive visualizations  
- Responsive, user-friendly recruiter/candidate interface  
- CSV upload, metric dashboard, and real-time feedback  

---

## ✅ Testing

| Test | Description              | Outcome                      |
|------|--------------------------|------------------------------|
| T01  | Upload valid dataset     | Generates predictions        |
| T02  | Trigger fairness metrics | Displays evaluation charts   |
| T03  | Upload empty dataset     | Error message displayed      |

---


