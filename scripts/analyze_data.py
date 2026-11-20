import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_folder / "data" / "titanic.csv")

columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df = df[columns].copy()
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

print("Dataset size:", df.shape)
print("\nSummary statistics:")
print(df.describe().round(2))

print("\nSurvival rate by passenger class (%):")
print((df.groupby("Pclass")["Survived"].mean() * 100).round(1))

print("\nSurvival rate by sex (%):")
print((df.groupby("Sex")["Survived"].mean() * 100).round(1)
      .rename(index={0: "Male", 1: "Female"}))

print("\nCorrelation with survival:")
print(df.corr(numeric_only=True)["Survived"].sort_values(ascending=False).round(3))
