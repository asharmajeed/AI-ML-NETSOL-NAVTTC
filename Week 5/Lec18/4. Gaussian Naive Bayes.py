from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = Path(__file__).parent / "Titanic-Dataset-Cleaned.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("=== Gaussian Naive Bayes Results ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred))

sample_passenger = X_test.iloc[[0]]
survival_prob = model.predict_proba(sample_passenger)[0][1]
print(f"Sample Passenger Survival Probability: {survival_prob:.2%}")
