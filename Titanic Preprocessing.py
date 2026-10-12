from pathlib import Path
import pandas as pd

file_path = Path(__file__).parent / "Titanic-Dataset.csv"
output_path = Path(__file__).parent / "Titanic-Dataset-Cleaned.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("------------------")
print("First 5 Rows:")
print(df.head())
print("------------------")

print("Missing Values Before Cleaning:")
print(df.isnull().sum())
print("------------------")

print("Duplicate Rows Count:", df.duplicated().sum())
print("------------------")

# FEATURE ENGINEERING & EXTRACTION

# Extract title from 'Name' (e.g., Mr, Mrs, Miss, Master, etc.)
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Group uncommon titles into 'Rare'
rare_titles = [
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", 
    "Major", "Rev", "Sir", "Jonkheer", "Dona"
]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Calculate total family size and an 'Is_Alone' flag
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

# Create 'Has_Cabin' flag (1 if cabin was recorded, 0 if missing)
df["Has_Cabin"] = df["Cabin"].notnull().astype(int)

# HANDLING MISSING VALUES

# 1. 'Age' (177 missing values):
# Impute missing Age based on the median Age within each Title group
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
# Fallback to overall median if any remaining
df["Age"] = df["Age"].fillna(df["Age"].median())

# 2. 'Embarked' (2 missing values):
# Impute missing Embarked with the mode ('S')
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)

# 3. 'Fare' (guard against any missing values):
# Impute with median Fare for the passenger's class (Pclass)
df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))

# ENCODING CATEGORICAL VARIABLES & DOMAIN INTERACTIONS

# Binary encoding for 'Sex': female = 0, male = 1
df["Sex"] = df["Sex"].map({"female": 0, "male": 1}).astype(int)

# Domain Feature: "Women and Children First" evacuation protocol
# Women (Sex == 0) or children aged 12 and below had highest lifeboat priority
df["Woman_or_Child"] = ((df["Sex"] == 0) | (df["Age"] <= 12)).astype(int)

# One-hot encoding for 'Embarked' and 'Title'
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True, dtype=int)

# DROP UNNECESSARY / HIGH-CARDINALITY COLUMNS

# Drop raw columns that are not useful for machine learning models or are now encoded
cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
df_cleaned = df.drop(columns=cols_to_drop)

# VERIFICATION & SAVE CLEANED DATASET
print("\nMissing Values After Cleaning:")
print(df_cleaned.isnull().sum())
print("------------------")

print("Cleaned Dataset Shape:", df_cleaned.shape)
print("------------------")
print("First 5 Rows of Cleaned Dataset:")
print(df_cleaned.head())
print("------------------")

df_cleaned.to_csv(output_path, index=False)
print(f"Cleaned dataset saved successfully to: {output_path.name}")
