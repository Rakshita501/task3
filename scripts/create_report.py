import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_folder / "data" / "titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["SexCode"] = df["Sex"].map({"male": 0, "female": 1})

total = len(df)
overall_survival = df["Survived"].mean() * 100
class_rates = (df.groupby("Pclass")["Survived"].mean() * 100).round(1)
sex_rates = (df.groupby("Sex")["Survived"].mean() * 100).round(1)
correlations = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "SexCode"]].corr()["Survived"]

report = f"""TITANIC DATA ANALYSIS REPORT

1. Dataset overview
The dataset contains {total} passengers.
Overall, {overall_survival:.1f}% survived.

2. Survival by passenger class
"""
for passenger_class, rate in class_rates.items():
    report += f"- Class {passenger_class}: {rate:.1f}% survived\n"

report += "\n3. Survival by sex\n"
for sex, rate in sex_rates.items():
    report += f"- {sex}: {rate:.1f}% survived\n"

report += "\n4. Correlation with survival\n"
report += f"- Sex (female coded 1): {correlations['SexCode']:.3f}\n"
report += f"- Passenger class: {correlations['Pclass']:.3f}\n"
report += f"- Fare: {correlations['Fare']:.3f}\n"
report += f"- Age: {correlations['Age']:.3f}\n"

report += """
5. Main insights
- Survival rates differed substantially by passenger class and sex.
- In this dataset, female passengers and passengers in higher classes had higher survival rates.
- Correlation shows association; it does not prove that a factor caused survival.

6. Charts
See outputs/charts for:
- survival_by_class.png
- survival_by_sex.png
- correlation_heatmap.png

7. Data note
Missing ages were filled with the median age for the summary and charts.
"""

report_folder = project_folder / "report"
report_folder.mkdir(parents=True, exist_ok=True)
report_file = report_folder / "findings_report.txt"
report_file.write_text(report, encoding="utf-8")

print("Report saved to:", report_file)
print(report)
