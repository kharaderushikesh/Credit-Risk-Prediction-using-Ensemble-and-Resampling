# 🧮 Credit Risk Prediction using Ensemble and Resampling

## Overview of Project

### Background
Credit risk prediction is a critical component in financial analytics. It determines whether a borrower is likely to default on a loan. The dataset used is inherently imbalanced, where the number of low-risk loans significantly exceeds high-risk ones.  
This project demonstrates two complementary approaches:
1. **Ensemble Learning Models**
2. **Resampling Techniques (RandomOverSampler, SMOTE, SMOTEENN)**

### Purpose
To design and evaluate multiple ensemble-based classification models and resampling strategies for improving predictive accuracy on imbalanced credit datasets.

---

## ⚙️ Project Structure
```
📁 Credit_Risk_Analysis/
│
├── credit_risk_ensemble.py         # Ensemble models pipeline
├── credit_risk_resampling.py       # Sampling + Ensemble pipeline
├── sample_loan_data_200.csv        # Sample dataset (200 records)
├── cleaned_credit_risk_ensemble.csv
├── cleaned_credit_risk_resampling.csv
└── README.md                       # Project documentation
```

---

## 📊 Project 1: Ensemble Modeling (`credit_risk_ensemble.py`)

### Models Used
- Random Forest Classifier  
- AdaBoost Classifier  
- Gradient Boosting Classifier  
- Voting Classifier (Soft Voting Ensemble)

### Workflow
1. Load and clean dataset  
2. Encode categorical columns using `LabelEncoder`  
3. Scale numeric features with `StandardScaler`  
4. Train all ensemble models  
5. Evaluate with confusion matrix, classification report, and accuracy  
6. Visualize accuracy comparison bar chart  

### Key Results
| Model | Accuracy | Precision (High Risk) | Recall (High Risk) |
|--------|-----------|----------------------|--------------------|
| Random Forest | ~0.89 | 0.87 | 0.91 |
| AdaBoost | ~0.84 | 0.83 | 0.86 |
| Gradient Boosting | ~0.88 | 0.85 | 0.90 |
| Voting Ensemble | **0.91** | **0.88** | **0.93** |

✅ The **Voting Classifier** ensemble performed best overall.

---

## 📊 Project 2: Resampling + Ensemble (`credit_risk_resampling.py`)

### Resampling Techniques
- RandomOverSampler  
- SMOTE (Synthetic Minority Oversampling Technique)  
- SMOTEENN (Hybrid SMOTE + Edited Nearest Neighbor)

### Models Used
- Random Forest  
- AdaBoost  
- Gradient Boosting  

### Workflow
1. Preprocess and encode dataset  
2. Apply each resampling technique  
3. Train and evaluate ensemble models  
4. Compare results across sampling methods  
5. Visualize class distributions and model accuracies  

### Key Insights
| Sampling Technique | Best Model | Accuracy | Recall (High Risk) |
|--------------------|-------------|-----------|--------------------|
| Original Data | Gradient Boosting | 0.86 | 0.84 |
| RandomOverSampler | Random Forest | 0.90 | 0.88 |
| SMOTE | Gradient Boosting | 0.91 | 0.90 |
| SMOTEENN | AdaBoost | **0.93** | **0.91** |

✅ **AdaBoost + SMOTEENN** produced the best balance between recall and accuracy.

---

## 🧮 Technologies Used
- Python 3.11+  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  
- imbalanced-learn  

---

## 📈 Visualizations
- Class distribution before and after resampling  
- Ensemble accuracy comparison chart  
- Confusion matrices and recall plots  

---

## 🧾 Summary
- Ensemble models enhance prediction stability.  
- Resampling balances class ratios and boosts minority-class accuracy.  
- **Voting Ensemble** (ensemble script) and **AdaBoost + SMOTEENN** (resampling script) provide the best outcomes.

💡 **Recommended Deployment:**  
Use **AdaBoost with SMOTEENN** for practical applications to achieve higher recall and lower false negatives in credit risk detection.

---

## 👨‍💻 Author
**Rushikesh Kharade**  
*B.Tech in Artificial Intelligence & Data Science*  
GitHub: [@kharaderushikesh](https://github.com/kharaderushikesh)
