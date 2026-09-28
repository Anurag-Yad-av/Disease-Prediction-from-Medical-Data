# CodeAlpha ML Internship — Task 4: Disease Prediction

**Objective:** Predict whether a breast tumor is classified as malignant or benign using structured medical measurements.

**Dataset:** Breast Cancer Wisconsin (Diagnostic) dataset available through scikit-learn.

**Algorithms:** Logistic Regression, SVM, Random Forest.

**Preprocessing:** 80/20 stratified split, median imputation, StandardScaler, and pipeline-based preprocessing.

**Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC and confusion matrices.

### Run
```bash
pip install -r requirements.txt
python disease_prediction.py
```

The program generates `model_results.csv` and confusion-matrix PNG files.

**Disclaimer:** Academic demonstration only; not a clinical diagnostic system.
