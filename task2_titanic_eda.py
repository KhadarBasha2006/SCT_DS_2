"""
SkillCraft Technology - Data Science Internship
Task 02: Data Cleaning & Exploratory Data Analysis (EDA)
Dataset: Titanic (Kaggle) - https://www.kaggle.com/c/titanic/data

Goal: Clean the dataset, explore relationships between variables, and
identify patterns/trends related to passenger survival.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
pd.set_option("display.width", 120)

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("train.csv")
print("Original shape:", df.shape)
print("\nMissing values before cleaning:\n", df.isnull().sum())

# =========================================================
# 2. DATA CLEANING
# =========================================================
df_clean = df.copy()

# Age: fill missing with the median age WITHIN each Pclass+Sex group
# (more accurate than a single global median, since age varies by class/sex)
df_clean["Age"] = df_clean.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: only 2 missing -> fill with the mode (most common port)
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# Cabin: ~77% missing, too sparse to impute meaningfully.
# Convert to a binary flag instead: did the passenger have a recorded cabin?
df_clean["HasCabin"] = df_clean["Cabin"].notna().astype(int)
df_clean.drop(columns=["Cabin"], inplace=True)

# Feature engineering: family size and "is alone"
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)

# Extract title from Name (Mr, Mrs, Miss, Master, etc.) - useful social signal
df_clean["Title"] = df_clean["Name"].str.extract(r",\s*([^\.]*)\.")
rare_titles = df_clean["Title"].value_counts()[df_clean["Title"].value_counts() < 10].index
df_clean["Title"] = df_clean["Title"].replace(rare_titles, "Rare")

# Age groups for easier categorical analysis
df_clean["AgeGroup"] = pd.cut(
    df_clean["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
)

print("\nMissing values after cleaning:\n", df_clean.isnull().sum())
df_clean.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")

# =========================================================
# 3. EXPLORATORY DATA ANALYSIS
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Titanic Dataset - Exploratory Data Analysis", fontsize=18, fontweight="bold")

# --- 3.1 Overall survival count ---
sns.countplot(data=df_clean, x="Survived", hue="Survived", palette=["#d62728", "#2ca02c"],
              legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Overall Survival Count")
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(["Died", "Survived"])
for c in axes[0, 0].containers:
    axes[0, 0].bar_label(c)

# --- 3.2 Survival rate by passenger class ---
sns.barplot(data=df_clean, x="Pclass", y="Survived", hue="Pclass", palette="Blues_d",
            legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Survival Rate by Passenger Class")
axes[0, 1].set_ylabel("Survival Rate")

# --- 3.3 Survival rate by sex ---
sns.barplot(data=df_clean, x="Sex", y="Survived", hue="Sex", palette=["#1f77b4", "#e377c2"],
            legend=False, ax=axes[0, 2])
axes[0, 2].set_title("Survival Rate by Sex")
axes[0, 2].set_ylabel("Survival Rate")

# --- 3.4 Age distribution split by survival ---
sns.histplot(data=df_clean, x="Age", hue="Survived", multiple="stack",
             palette=["#d62728", "#2ca02c"], bins=30, ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution by Survival")
axes[1, 0].legend(labels=["Survived", "Died"])

# --- 3.5 Fare distribution split by survival (log scale, fare is skewed) ---
sns.boxplot(data=df_clean, x="Survived", y="Fare", hue="Survived",
            palette=["#d62728", "#2ca02c"], legend=False, ax=axes[1, 1])
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Fare Distribution by Survival (log scale)")
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(["Died", "Survived"])

# --- 3.6 Survival rate by class AND sex (the famous "women and children" effect) ---
sns.barplot(data=df_clean, x="Pclass", y="Survived", hue="Sex", palette=["#e377c2", "#1f77b4"],
            ax=axes[1, 2])
axes[1, 2].set_title("Survival Rate by Class & Sex")
axes[1, 2].set_ylabel("Survival Rate")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("titanic_eda_overview.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nSaved titanic_eda_overview.png")

# =========================================================
# 4. CORRELATION HEATMAP
# =========================================================
plt.figure(figsize=(9, 7))
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"]
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Correlation Heatmap - Titanic Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("titanic_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved titanic_correlation_heatmap.png")

# =========================================================
# 5. KEY FINDINGS (printed summary)
# =========================================================
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

overall_rate = df_clean["Survived"].mean()
print(f"Overall survival rate: {overall_rate:.1%}")

by_sex = df_clean.groupby("Sex")["Survived"].mean()
print(f"\nSurvival rate by sex:\n{by_sex}")

by_class = df_clean.groupby("Pclass")["Survived"].mean()
print(f"\nSurvival rate by class:\n{by_class}")

by_alone = df_clean.groupby("IsAlone")["Survived"].mean()
print(f"\nSurvival rate, alone vs with family:\n{by_alone}")

corr_fare_survival = df_clean["Fare"].corr(df_clean["Survived"])
print(f"\nCorrelation between Fare and Survival: {corr_fare_survival:.3f}")
