# Handling Missing Data in Machine Learning

A comparison of imputation strategies (Median, KNN, MICE) and missingness patterns (MCAR vs MNAR) on a clinical tabular dataset.

## Research question
How much does missing data reduce model accuracy, and does the choice of imputation method or the missingness pattern change the answer?

## Data
Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn: 569 patients, 6 measurements used (mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness). Target: malignant vs benign.

## Method
1. Start from the complete dataset (the "truth").
2. Delete 10% to 50% of the values on purpose, either completely at random (MCAR) or depending on the value itself (MNAR).
3. Fill the gaps with Median, KNN or MICE imputation.
4. Train a Random Forest and measure AUC with 5-fold cross-validation. Imputation is inside a scikit-learn Pipeline, so there is no data leakage.
5. Compare training vs testing performance using an 80/20 split, with confusion matrices and ROC curves.

## Key results

**Complete-data AUC: 0.981**

AUC by imputation method and % of values missing (5-fold cross-validation):

| Missing % | Median | KNN | MICE |
|---|---|---|---|
| 10 | 0.979 | 0.971 | 0.977 |
| 20 | 0.972 | 0.969 | 0.975 |
| 30 | 0.961 | 0.946 | 0.966 |
| 40 | 0.956 | 0.949 | 0.957 |
| 50 | 0.944 | 0.922 | 0.940 |
| **Average** | 0.962 | 0.951 | 0.963 |

MCAR vs MNAR at 30% missing:

| Method | Random (MCAR) | Value-dependent (MNAR) |
|---|---|---|
| Median | 0.961 | 0.958 |
| KNN | 0.946 | 0.944 |
| MICE | 0.966 | 0.960 |

Training vs testing (30% missing, 80/20 split):

| Model | Train AUC | Test AUC |
|---|---|---|
| Complete data | 1.000 | 0.980 |
| Median | 1.000 | 0.938 |
| KNN | 1.000 | 0.905 |
| MICE | 1.000 | 0.951 |

## Findings
- Accuracy falls as more data goes missing. AUC dropped from 0.981 (complete data) to between 0.922 and 0.944 at 50% missing.
- KNN imputation had the lowest AUC at every missing level and the lowest test AUC (0.905).
- Median and MICE performed very similarly (average AUC 0.962 vs 0.963). The gap is too small to call a winner from this experiment.
- The missingness pattern had a small effect here: MNAR was at most 0.006 AUC lower than MCAR at 30% missing.
- Training AUC was 1.000 for every model, because a Random Forest can memorise training data. The test AUC is the honest measure of performance.

![AUC vs missing](figures/cell09_auc_vs_missing.png)
![ROC curves](figures/cell15_roc.png)

## Limitations
- One small dataset and 6 features only.
- Missing values were simulated, not real.
- One model (Random Forest); results may differ for other models.
- One random missing-data mask per level (no repeated runs), so differences of about 0.005 AUC, such as Median vs MICE, should not be treated as meaningful.
- The train/test split uses only about 114 test patients and one split, so those scores are noisy.

## Future work
- Repeat with several random seeds and report variability.
- Test more datasets and models.
- Add calibration and explainability analysis.
- Extend to missing information in medical images.

## Run it
Open `missing_data_imputation.ipynb` in Google Colab and run the cells from top to bottom.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
