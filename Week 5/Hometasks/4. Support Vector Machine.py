from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

print("=== Comparing SVM Kernels ===")
best_model = None
best_acc = 0
best_kernel = ""

for kernel in ["linear", "rbf"]:
    for c in [0.5, 1.0, 2.0]:
        model = SVC(kernel=kernel, C=c, random_state=42)
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"SVM {kernel:<7} (C={c:>3}): Accuracy = {acc:.4f} ({acc * 100:.2f}%)")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_kernel = f"{kernel} (C={c})"

print("-----------------------------------")
print(f"Best SVM Config: {best_kernel} with Accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")
print("-----------------------------------")

pred = best_model.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Not Spam (Ham)", "Spam"]))

sample_email = X_test_scaled[[0]]
actual_label = "Spam" if y_test.iloc[0] == 1 else "Not Spam (Ham)"
pred_val = best_model.predict(sample_email)[0]
pred_label = "Spam" if pred_val == 1 else "Not Spam (Ham)"
decision_score = best_model.decision_function(sample_email)[0]

print(f"Sample Email - Actual: {actual_label} | Predicted: {pred_label} (Confidence Score: {decision_score:.3f})")
