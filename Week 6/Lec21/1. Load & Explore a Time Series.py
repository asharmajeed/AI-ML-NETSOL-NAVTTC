import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "shop_sales_daily.csv"

df = pd.read_csv(file_path, parse_dates=["date"])

print(df.head(3))

print(
    "From", df["date"].min().date(), "to", df["date"].max().date(), "|", len(df), "days"
)

df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

print("\nAverage sales by day of week:")
print(df.groupby("dayofweek")["sales"].mean().round())

print("\nAverage sales by year:")
print(df.groupby(df["date"].dt.year)["sales"].mean().round())
