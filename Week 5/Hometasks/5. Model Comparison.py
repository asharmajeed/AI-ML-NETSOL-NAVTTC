from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

file_path = Path(__file__).parent / "emails.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["spam"])
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Gaussian Naive Bayes": (GaussianNB(), False),
    "Logistic Regression": (LogisticRegression(max_iter=1000, C=1.0, random_state=42), True),
    "K-Nearest Neighbours (k=5)": (KNeighborsClassifier(n_neighbors=5), True),
    "Support Vector Machine (RBF)": (SVC(kernel="rbf", C=1.0, random_state=42), True),
}

results = []

print("=" * 65)
print("     SPAM EMAIL FILTERING - SUPERVISED LEARNING COMPARISON")
print("=" * 65)

for name, (model, use_scaled) in models.items():
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test

    model.fit(X_tr, y_train)
    pred = model.predict(X_te)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    results.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "Recall (%)": round(rec * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "False Positives (Ham -> Spam)": cm[0][1],
        "False Negatives (Spam -> Ham)": cm[1][0],
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))
print("=" * 65)

best_acc_model = results_df.loc[results_df["Accuracy (%)"].idxmax()]["Model"]
best_prec_model = results_df.loc[results_df["Precision (%)"].idxmax()]["Model"]

print(f"\nBest Overall Accuracy : {best_acc_model}")
print(f"Lowest False Alarm Rate: {best_prec_model} (Crucial for preventing legitimate emails from landing in Spam)")
