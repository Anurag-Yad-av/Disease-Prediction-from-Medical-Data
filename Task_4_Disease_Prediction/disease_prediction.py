import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay

# CodeAlpha ML Internship - Task 4: Disease Prediction
# Educational project only; not a clinical diagnostic system.

data = load_breast_cancer(as_frame=True)
X, y = data.data.copy(), data.target.copy()

print("Dataset shape:", X.shape)
print("Class distribution:", y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
}

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, target_names=["Malignant", "Benign"], zero_division=0))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Malignant", "Benign"]
    )
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(name.lower().replace(" ", "_") + "_confusion_matrix.png", dpi=200)
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)
print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))
