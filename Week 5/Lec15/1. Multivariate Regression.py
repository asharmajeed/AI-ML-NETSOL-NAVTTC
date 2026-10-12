from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

file_path = Path(__file__).parent / "students_week5.csv"

df = pd.read_csv(file_path)
features = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "sleep_hours",
    "assignments_done",
    "internet_access",
    "extra_tuition",
    "part_time_job",
    "parent_education",
]

X = df[features]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("R2 :", round(r2_score(y_test, pred), 3))
print("RMSE :", round(mean_squared_error(y_test, pred) ** 0.5, 2))
print(pd.Series(model.coef_, index=features).round(2))
