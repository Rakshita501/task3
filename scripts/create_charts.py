import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_folder / "data" / "titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
sns.set_theme(style="whitegrid")
charts_folder = project_folder / "outputs" / "charts"
charts_folder.mkdir(parents=True, exist_ok=True)

# Survival rate by passenger class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Pclass",
            legend=False, palette="Blues")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(charts_folder / "survival_by_class.png", dpi=150)
plt.close()

# Survival rate by sex
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Sex", y="Survived", hue="Sex",
            legend=False, palette="Set2")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(charts_folder / "survival_by_sex.png", dpi=150)
plt.close()

# Correlation heatmap
analysis_df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()
analysis_df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
plt.figure(figsize=(8, 6))
sns.heatmap(analysis_df.corr(numeric_only=True), annot=True,
            cmap="coolwarm", center=0, fmt=".2f")
plt.title("Titanic Feature Correlations")
plt.tight_layout()
plt.savefig(charts_folder / "correlation_heatmap.png", dpi=150)
plt.close()

print("Charts saved in:", charts_folder)
