from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

file_path = Path(__file__).parent / "students_week5.csv"

df = pd.read_csv(file_path)

X = df.drop(columns=["student_id", "final_score", "result", "performance"])
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for degree in [1, 2, 3]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"Degree {degree}: train R2 = {train_r2:.3f} | test R2 = {test_r2:.3f}")
