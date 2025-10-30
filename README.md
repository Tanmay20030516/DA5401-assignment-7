# DA5401 A7: Multi-Class Model Selection using ROC and  Precision-Recall Curves

**Name:** Tanmay Gawande  
**Roll Number:** DA25M030

## Project Directory
```bash
DA5401-assignment-7/
├── da25m030-assignment-7-solution.ipynb    # Main Jupyter Notebook
├── requirements.txt                        # Project requirements
├── sat.doc                                 # Information about dataset
├── sat.trn                                 # Train dataset
├── sat.tst                                 # Test dataset
└── README.md                               # Project documentation
```

## **Overview**

This project conducts a comprehensive evaluation of 10+ classification models to solve a multi-class problem. The analysis moves beyond simple accuracy to focus on robust metrics suitable for imbalanced data, including **ROC-AUC** (Receiver Operating Characteristic Area Under the Curve) and **mAP** (Mean Average Precision, from Precision-Recall curves).

The core of the project involves:
1. Training a suite of models (from `LogisticRegression` to `XGBClassifier`).
2. Evaluating them on `weighted_f1`, `ROC-AUC`, and `mAP`.
3. Analyzing the **rank correlation (Spearman's $\rho$)** between these metrics to understand their agreement.
4. Attempting **per-class threshold tuning** to optimize F1-scores.
5. Recommending a final model based on the most reliable performance indicators.

## **Final Model Performance Metrics**

The table below summarizes the performance of all models on the test set *before* threshold tuning (as tuning was found to be ineffective). The models are sorted by **mAP (Mean Average Precision)**, which is the most reliable metric for this imbalanced classification task.

| **Model** | **ROC-AUC** | **mAP** | **accuracy** | **weighted_f1** | **macro_f1** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBClassifier** | 0.99015 | 0.95094 | 0.9050 | 0.90296 | 0.88858 |
| **RandomForestClassifier** | 0.98966 | 0.95009 | 0.9055 | 0.90358 | 0.89140 |
| **GradientBoostingClassifier** | 0.98679 | 0.93642 | 0.8920 | 0.89059 | 0.87489 |
| **KNeighborsClassifier** | 0.97857 | 0.92167 | 0.9045 | 0.90375 | 0.89155 |
| **SVC** | 0.98518 | 0.91740 | 0.8955 | 0.89249 | 0.87692 |
| **LogisticRegression** | 0.97574 | 0.87106 | 0.8395 | 0.82960 | 0.79705 |
| **MLPClassifier** | 0.97309 | 0.86228 | 0.8350 | 0.82493 | 0.79203 |
| **GaussianNB** | 0.95535 | 0.81045 | 0.7965 | 0.80358 | 0.78328 |
| **DecisionTreeClassifier** | 0.90029 | 0.73735 | 0.8510 | 0.85141 | 0.83227 |
| **DummyClassifier** | 0.50000 | 0.16667 | 0.2305 | 0.08636 | 0.06244 |
| **DummyClassifier_v2** | 0.50000 | 0.16667 | 0.1735 | 0.17906 | 0.16734 |

## **Key Analysis & Conclusion**

### 1. Rank Correlation Analysis

A **Spearman Rank Correlation** was performed to compare the three key metrics.

* **ROC-AUC vs. mAP ($\rho = 0.918$):** An extremely strong positive correlation. This shows that both metrics were in high agreement on which models were best, strengthening our confidence in them.
* **F1 vs. ROC-AUC ($\rho = -0.273$):** A weak *negative* correlation. This was a critical finding, proving that the models that scored highest on the `weighted_f1` (like `KNeighborsClassifier`) were *not* the best when evaluated across all thresholds.

### 2. Threshold Tuning Ineffective

Per-class threshold tuning was performed to maximize F1-scores, but it provided no significant performance boost. This was because:
* Top models like `XGBClassifier` were already highly accurate, so their default `argmax()` prediction was already the correct choice.
* The "optimal" thresholds were slightly overfit to the training set and didn't generalize.

### 3. Final Recommendation: `XGBClassifier`

The **`XGBClassifier`** is the recommended model for this task.

While `KNeighborsClassifier` had a slightly higher `weighted_f1` score, the rank analysis proved `F1-score` was a misleading metric. The `XGBClassifier` was the undisputed **#1 ranked model** on the two most robust and important metrics: **`mAP` (0.951)** and **`ROC-AUC` (0.990)**. This demonstrates it has the most superior, stable, and well-calibrated performance across all possible decision thresholds.

---
