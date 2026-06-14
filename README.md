# Employee Attrition Prediction System

##  Overview

Employee attrition — the voluntary or involuntary departure of employees — is one of the most costly challenges organizations face, leading to increased hiring expenses, loss of institutional knowledge, and reduced team productivity.

This project builds an end-to-end **supervised machine learning pipeline** to predict whether an employee is likely to leave the organization, using historical HR data. Five classification algorithms are benchmarked and evaluated to identify the most effective model for this task.



##  Objectives

- Analyze and preprocess real-world HR data to uncover attrition patterns
- Engineer meaningful features and handle class imbalance using SMOTE
- Train, tune, and benchmark five ML classification models
- Evaluate model performance using Accuracy, F1-Score, Precision, Recall, and ROC-AUC
- Support data-driven HR decision-making through predictive insights


##  Dataset

|             Property            |             Details               |
|---|---|---|
| Source                          |    IBM HR Analytics Employee Attrition Dataset (Kaggle)            |
| Total Records                   |    1,470 employees                    |
| Features (raw)                  |    35 attributes                      |
| Features (after selection) |    29 attributes                    |
| Target Variable                      |    Attrition (Yes = 1, No = 0)  |
| Class Distribution                      |    ~84% No Attrition, ~16% Attrition (imbalanced)  |


**Key Features Used:**
- Demographics: Age, Gender, MaritalStatus
- Job Info: JobRole, JobInvolvement, JobSatisfaction, OverTime
- Compensation: MonthlyIncome, PercentSalaryHike
- Experience: TotalWorkingYears, YearsAtCompany, YearsInCurrentRole



##  Project Workflow

```
Raw Data → Feature Selection → Preprocessing → EDA → SMOTE → Model Training → Evaluation → Prediction
```

1. **Data Collection** — Loaded IBM HR dataset (CSV)
2. **Feature Selection** — Dropped 15 irrelevant/redundant columns
3. **Preprocessing** — Label encoding, one-hot encoding, standard scaling
4. **Exploratory Data Analysis (EDA)** — Distribution plots, correlation analysis, attrition patterns
5. **Class Imbalance Handling** — Applied SMOTE to oversample minority class
6. **Model Training** — Trained 5 classification algorithms
7. **Hyperparameter Tuning** — GridSearchCV for Decision Tree and SVM
8. **Evaluation** — Compared models on multiple metrics
9. **Inference** — Tested real-world sample predictions

---

##  Models Implemented
  
|             Model            |             Key Technique                |                    Notes                  |
|---|---|---|
| Logistic Regression          |    Baseline linear classifier            |     Fast, interpretable                   |
| Decision Tree                |    GridSearchCV tuned                    |     max_depth, min_samples tuned          |
| Balanced Random Forest       |    Ensemble + SMOTE                      |     Handles class imbalance natively      |
| Support Vector Machine (SVM) |    GridSearchCV tuned                    |     RBF & Linear kernels compared         |
| XGBoost                      |    Gradient Boosting + scale_pos_weight  |     Best overall performance              |

---

##  Tools & Technologies

|        Category        |                             Tools                           |
|---|---|
| Language               |       Python 3.10                                           |
| ML Framework           |       Scikit-learn, XGBoost, Imbalanced-learn               |
| Data Processing        |       Pandas, NumPy                                         |
| Visualization          |       Matplotlib, Seaborn                                   |
| Imbalance Handling     |       SMOTE (Synthetic Minority Over-sampling Technique)    |
| Hyperparameter Tuning  |       GridSearchCV (5-fold cross-validation)                |
| Model Persistence      |       Joblib                                                |
| Environment            |       Google Colab                                          |

---

##  Results

### XGBoost — Best Performing Model

|            Metric             |       Score       |
|---|---|
| **Accuracy**                  |     **84.35%** |
| **Weighted F1-Score**         |     **0.85** |
| **Weighted Precision**        |     **0.85** |
| **Weighted Recall**           |     **0.84** |

### Classification Report (XGBoost)

```
              precision    recall  f1-score   support

           0       0.91      0.91      0.91       255
           1       0.41      0.44      0.42        39

    accuracy                           0.84       294
   macro avg       0.66      0.67      0.67       294
weighted avg       0.85      0.84      0.85       294
```

### Confusion Matrix

```
[[231  24]
 [ 22  17]]
```

> **Note:** The lower F1 for Class 1 (Attrition = Yes) reflects the inherent class imbalance in the dataset (~16% attrition rate). SMOTE was applied to partially mitigate this — future work will explore advanced resampling and cost-sensitive learning.

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/Kabila-US/Employee-Attrition-Prediction-System.git
```

2. Open the notebook in Google Colab or Jupyter:
```
Employee_Attrition_Prediction_System.ipynb
```

3. Upload the dataset file `ATTRITION.csv` when prompted

4. Run all cells sequentially to reproduce results

---

##  Future Enhancements

- Deployment as an interactive web dashboard using Streamlit or Flask
- Real-time attrition monitoring with live HR data integration
- Explainability layer using SHAP values to interpret individual model predictions
- Extending the pipeline to support multi-class attrition risk levels (Low / Medium / High)
- Integration with HR tools like SAP or Workday for enterprise-level deployment

---

##  Conclusion

This project demonstrates the practical application of machine learning in **Human Resource Analytics**. By systematically benchmarking five classification algorithms, handling real-world class imbalance, and applying hyperparameter tuning, the pipeline achieves **84.35% accuracy** with XGBoost as the best-performing model — providing a foundation for data-driven employee retention strategies.

---

##  Author

**Kabila U S**
- LinkedIn: linkedin.com/in/kabila-u-s
- GitHub: github.com/Kabila-US
