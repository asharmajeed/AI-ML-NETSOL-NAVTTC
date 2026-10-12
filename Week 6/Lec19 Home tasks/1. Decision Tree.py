from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = Path(__file__).parent / "mushrooms.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("Target Class Distribution:\n", df["type"].value_counts())
print("-" * 55)

df = df.drop(columns=["veil_type"])
df["stalk_root"] = df["stalk_root"].replace("?", "missing")

y = (df["type"] == "p").astype(int)
X_raw = df.drop(columns=["type"])

X = pd.get_dummies(X_raw, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== Evaluating Decision Tree at Different Depths ===")
for depth in [3, 5, 7, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)
    depth_name = f"max_depth={depth}" if depth is not None else "max_depth=None (Full)"
    print(
        f"{depth_name:<22}: Train = {train_acc * 100:.2f}% | Test = {test_acc * 100:.2f}%"
    )

model = DecisionTreeClassifier(max_depth=7, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("\n" + "-" * 55)
print(f"      DECISION TREE FINAL ACCURACY: {accuracy * 100:.2f}%")
print("-" * 55)

print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Edible (0)", "Poisonous (1)"]))

importance = pd.Series(model.feature_importances_, index=X.columns)
top_features = importance[importance > 0].sort_values(ascending=False)
print("Top Decision Features (Importance > 0):")
print(top_features.round(4))

print("\n=== Decision Tree Logic (Top Rules) ===")
small_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(export_text(small_tree, feature_names=list(X.columns)))
