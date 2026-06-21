# SCT_DS_2 — Task 02: Data Cleaning & Exploratory Data Analysis (EDA)

**Internship:** SkillCraft Technology – Data Science Track
**Task:** Perform data cleaning and EDA on a dataset of your choice (Titanic dataset from Kaggle).

## 🎯 Objective
Clean the Titanic dataset, handle missing data, engineer useful features, and explore
relationships between passenger attributes and survival to identify patterns and trends.

## 🗂️ Dataset
- Source: [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic/data)
- Files used (included in this repo):
  - `train.csv` — 891 passengers with known survival outcome
  - `test.csv` — 418 passengers (no survival label, for reference)
  - `gender_submission.csv` — Kaggle's sample submission format
- Output: `titanic_cleaned.csv` — the cleaned, feature-engineered dataset produced by the script

## 🛠️ Tools Used
- Python 3, pandas, numpy
- matplotlib, seaborn (visualization)

## 🧹 Data Cleaning Steps
1. **Age** (177 missing): imputed using the median age within each `Pclass` + `Sex` group, rather than a single global median — more representative since age varies meaningfully across class and sex.
2. **Embarked** (2 missing): filled with the mode (most frequent port).
3. **Cabin** (687 missing, ~77%): too sparse to impute reliably, so converted to a binary `HasCabin` flag instead of dropping the signal entirely.
4. **Feature engineering**:
   - `FamilySize` = SibSp + Parch + 1
   - `IsAlone` = 1 if traveling solo
   - `Title` extracted from passenger name (Mr, Mrs, Miss, Master, Rare)
   - `AgeGroup` binned into Child / Teen / Young Adult / Adult / Senior

## 📊 Exploratory Analysis
The script (`task2_titanic_eda.py`) produces:
- `titanic_eda_overview.png` — 6-panel overview: survival counts, survival rate by class, by sex, age distribution by survival, fare distribution by survival, and the combined class×sex breakdown
- `titanic_correlation_heatmap.png` — correlation matrix across all numeric/engineered features

## 🔑 Key Findings
- **Overall survival rate:** 38.4%
- **Sex was the strongest predictor:** women survived at 74.2% vs. men at 18.9%
- **Class mattered a lot:** 1st class 63.0% survival → 2nd class 47.3% → 3rd class 24.2%
- **"Women and children first" held up in the data:** in every class, female survival rate far exceeds male survival rate (e.g. 1st class women ~97% vs. 1st class men ~37%)
- **Traveling with family helped:** passengers not traveling alone survived at 50.6% vs. 30.4% for solo travelers
- **Fare correlates with survival** (r = 0.26) — largely because higher fares correlate with higher class and `HasCabin` status

## ▶️ How to run
```bash
pip install pandas numpy matplotlib seaborn
python task2_titanic_eda.py
```
Run from the same folder as `train.csv` — the script reads it directly and writes all outputs (cleaned CSV + 2 charts) into that same folder.

---
*Part of the SkillCraft Technology Data Science Internship — Task 2 of 4.*
