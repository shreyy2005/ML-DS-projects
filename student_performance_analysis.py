# ============================================================
#  STUDENT PERFORMANCE ANALYSIS — Data Science Project
#  Author: Shrey Dukare | BSIOTR, Pune | B.E. Computer Engineering
#  Tools: Python, Pandas, Matplotlib, Seaborn, Scipy (Hypothesis Testing)
#  Copy-paste this entire file into ONE Google Colab cell
# ============================================================

# ---------- STEP 1: Install & Import ----------
import subprocess
subprocess.run(["pip", "install", "scipy seaborn --quiet"], shell=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')
print("✅ Libraries imported!")

# ---------- STEP 2: Load Dataset ----------
# Using the Student Performance dataset from UCI ML Repository
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv"
df = pd.read_csv(url, sep=';')

print(f"✅ Dataset loaded: {df.shape[0]} students, {df.shape[1]} features")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# ---------- STEP 3: Basic Exploration ----------
print("\n" + "="*55)
print("DATASET OVERVIEW")
print("="*55)
print(f"Shape       : {df.shape}")
print(f"Missing vals: {df.isnull().sum().sum()}")
print(f"\nFinal Grade (G3) Stats:")
print(df['G3'].describe().round(2))

# Create a Pass/Fail column (G3 >= 10 is pass in Portuguese system)
df['result'] = df['G3'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
print(f"\nPass/Fail split:")
print(df['result'].value_counts())

# ---------- STEP 4: EDA — Grade Distribution ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Student Performance — Exploratory Data Analysis', fontsize=15, fontweight='bold')

# Final grade distribution
axes[0, 0].hist(df['G3'], bins=20, color='#3498db', edgecolor='black', alpha=0.8)
axes[0, 0].axvline(df['G3'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['G3'].mean():.1f}")
axes[0, 0].set_title('Final Grade (G3) Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Grade (0–20)')
axes[0, 0].set_ylabel('Number of Students')
axes[0, 0].legend()

# Pass vs Fail pie chart
colors = ['#2ecc71', '#e74c3c']
df['result'].value_counts().plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%',
                                  colors=colors, startangle=90, explode=[0, 0.05])
axes[0, 1].set_title('Pass vs Fail Ratio', fontweight='bold')
axes[0, 1].set_ylabel('')

# Study time vs G3
study_grade = df.groupby('studytime')['G3'].mean()
axes[1, 0].bar(study_grade.index, study_grade.values, color='#9b59b6', edgecolor='black')
axes[1, 0].set_title('Avg Final Grade by Study Time', fontweight='bold')
axes[1, 0].set_xlabel('Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)')
axes[1, 0].set_ylabel('Average Grade')
for i, v in enumerate(study_grade.values):
    axes[1, 0].text(study_grade.index[i], v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')

# Absences vs G3 scatter
axes[1, 1].scatter(df['absences'], df['G3'], alpha=0.4, color='#e67e22', edgecolor='white', s=50)
m, b = np.polyfit(df['absences'], df['G3'], 1)
axes[1, 1].plot(df['absences'], m * df['absences'] + b, color='red', linewidth=2, label='Trend')
axes[1, 1].set_title('Absences vs Final Grade', fontweight='bold')
axes[1, 1].set_xlabel('Number of Absences')
axes[1, 1].set_ylabel('Final Grade (G3)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("💡 Insight: More study time → higher grades. More absences → lower grades!")

# ---------- STEP 5: Correlation Heatmap ----------
numeric_cols = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True,
            annot_kws={'size': 10})
plt.title('Correlation Heatmap — Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("💡 Insight: G1 and G2 (mid-term grades) are the strongest predictors of G3!")

# ---------- STEP 6: Hypothesis Testing ----------
print("\n" + "="*55)
print("HYPOTHESIS TESTING")
print("="*55)

# --- TEST 1: T-Test — Does internet access affect grades? ---
print("\n📌 TEST 1: T-Test — Does internet access affect final grade?")
print("   H0 (Null): Internet access has NO effect on grades")
print("   H1 (Alt) : Internet access DOES affect grades")
internet_yes = df[df['internet'] == 'yes']['G3']
internet_no  = df[df['internet'] == 'no']['G3']
t_stat, p_val = stats.ttest_ind(internet_yes, internet_no)
print(f"   Mean grade WITH internet   : {internet_yes.mean():.2f}")
print(f"   Mean grade WITHOUT internet: {internet_no.mean():.2f}")
print(f"   T-statistic: {t_stat:.4f}")
print(f"   P-value    : {p_val:.4f}")
print(f"   Result: {'✅ REJECT H0 — Internet access significantly affects grades!' if p_val < 0.05 else '❌ FAIL TO REJECT H0 — No significant effect found'}")

# --- TEST 2: Chi-Square — Is gender associated with pass/fail? ---
print("\n📌 TEST 2: Chi-Square Test — Is gender associated with Pass/Fail?")
print("   H0: Gender and exam result are INDEPENDENT")
print("   H1: Gender and exam result are DEPENDENT (associated)")
contingency = pd.crosstab(df['sex'], df['result'])
print(f"\n   Contingency Table:\n{contingency}")
chi2, p_chi, dof, expected = chi2_contingency(contingency)
print(f"\n   Chi-Square statistic: {chi2:.4f}")
print(f"   Degrees of freedom  : {dof}")
print(f"   P-value             : {p_chi:.4f}")
print(f"   Result: {'✅ REJECT H0 — Gender IS associated with result!' if p_chi < 0.05 else '❌ FAIL TO REJECT H0 — Gender is NOT significantly associated with result'}")

# --- TEST 3: ANOVA — Does parental education level affect grades? ---
print("\n📌 TEST 3: One-Way ANOVA — Does mother's education level affect grades?")
print("   H0: All education groups have the SAME mean grade")
print("   H1: At least one group has a DIFFERENT mean grade")
groups = [df[df['Medu'] == level]['G3'].values for level in sorted(df['Medu'].unique())]
f_stat, p_anova = f_oneway(*groups)
print(f"\n   Mean grades by mother's education level:")
for level in sorted(df['Medu'].unique()):
    label = {0:'None', 1:'Primary', 2:'Middle', 3:'Secondary', 4:'Higher'}[level]
    print(f"   Level {level} ({label}): {df[df['Medu']==level]['G3'].mean():.2f}")
print(f"\n   F-statistic: {f_stat:.4f}")
print(f"   P-value    : {p_anova:.4f}")
print(f"   Result: {'✅ REJECT H0 — Mother education level SIGNIFICANTLY affects grades!' if p_anova < 0.05 else '❌ FAIL TO REJECT H0 — No significant difference found'}")

# --- TEST 4: Bayes Theorem — P(Pass | Studies > 2hrs) ---
print("\n📌 TEST 4: Bayes' Theorem — P(Pass | Study Time > 2 hours)")
print("   Formula: P(Pass|Study>2) = P(Study>2|Pass) × P(Pass) / P(Study>2)")
P_pass          = len(df[df['result'] == 'Pass']) / len(df)
P_study_high    = len(df[df['studytime'] >= 3]) / len(df)
P_study_given_pass = len(df[(df['studytime'] >= 3) & (df['result'] == 'Pass')]) / len(df[df['result'] == 'Pass'])
P_pass_given_study = (P_study_given_pass * P_pass) / P_study_high
print(f"\n   P(Pass)                    : {P_pass:.4f} ({P_pass*100:.1f}%)")
print(f"   P(Study > 2hrs)            : {P_study_high:.4f} ({P_study_high*100:.1f}%)")
print(f"   P(Study > 2hrs | Pass)     : {P_study_given_pass:.4f} ({P_study_given_pass*100:.1f}%)")
print(f"\n   ✅ P(Pass | Study > 2hrs)  : {P_pass_given_study:.4f} ({P_pass_given_study*100:.1f}%)")
print(f"   💡 Students who study >2hrs/week have a {P_pass_given_study*100:.1f}% chance of passing!")

# ---------- STEP 7: Visualize Hypothesis Test Results ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Hypothesis Testing — Visual Results', fontsize=14, fontweight='bold')

# T-Test: Internet vs Grade
data_plot = [internet_yes.values, internet_no.values]
bp = axes[0].boxplot(data_plot, patch_artist=True, labels=['With Internet', 'Without Internet'])
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[0].set_title(f'T-Test: Internet vs Grade\np = {p_val:.4f} {"✅ Significant" if p_val < 0.05 else "❌ Not Significant"}', fontweight='bold')
axes[0].set_ylabel('Final Grade (G3)')

# Chi-Square: Gender vs Result
contingency.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#2ecc71'], edgecolor='black', rot=0)
axes[1].set_title(f'Chi-Square: Gender vs Result\np = {p_chi:.4f} {"✅ Significant" if p_chi < 0.05 else "❌ Not Significant"}', fontweight='bold')
axes[1].set_xlabel('Gender (F = Female, M = Male)')
axes[1].set_ylabel('Count')
axes[1].legend(['Fail', 'Pass'])

# ANOVA: Mother education vs Grade
edu_labels = {0:'None', 1:'Primary', 2:'Middle', 3:'Secondary', 4:'Higher'}
means = [df[df['Medu'] == l]['G3'].mean() for l in sorted(df['Medu'].unique())]
colors_anova = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
bars = axes[2].bar([edu_labels[l] for l in sorted(df['Medu'].unique())], means,
                    color=colors_anova, edgecolor='black')
axes[2].set_title(f'ANOVA: Mother Education vs Grade\np = {p_anova:.4f} {"✅ Significant" if p_anova < 0.05 else "❌ Not Significant"}', fontweight='bold')
axes[2].set_xlabel("Mother's Education Level")
axes[2].set_ylabel('Avg Final Grade')
for bar, m in zip(bars, means):
    axes[2].text(bar.get_x() + bar.get_width()/2, m + 0.1, f'{m:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('hypothesis_tests.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------- STEP 8: Key Factors Affecting Performance ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Study time vs pass rate
study_pass = df.groupby('studytime').apply(lambda x: (x['result'] == 'Pass').mean() * 100)
axes[0].bar(study_pass.index, study_pass.values, color='#9b59b6', edgecolor='black')
axes[0].set_title('Pass Rate by Study Time', fontweight='bold')
axes[0].set_xlabel('Study Time (1=<2hrs → 4=>10hrs)')
axes[0].set_ylabel('Pass Rate (%)')
axes[0].set_ylim(0, 110)
for i, v in zip(study_pass.index, study_pass.values):
    axes[0].text(i, v + 1, f'{v:.0f}%', ha='center', fontweight='bold')

# Failures vs G3
fail_grade = df.groupby('failures')['G3'].mean()
axes[1].bar(fail_grade.index, fail_grade.values, color='#e74c3c', edgecolor='black')
axes[1].set_title('Avg Grade vs Past Failures', fontweight='bold')
axes[1].set_xlabel('Number of Past Class Failures')
axes[1].set_ylabel('Average Final Grade')
for i, v in zip(fail_grade.index, fail_grade.values):
    axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('key_factors.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------- STEP 9: Summary ----------
print("\n" + "="*55)
print("PROJECT SUMMARY")
print("="*55)
print(f"""
Dataset  : 395 Portuguese high school students
Target   : Final Math grade (G3, scale 0–20)
Pass rate: {P_pass*100:.1f}%

Key Findings:
1. T-Test    → Internet access {'DOES' if p_val < 0.05 else 'does NOT'} significantly affect grades (p={p_val:.4f})
2. Chi-Square→ Gender {'IS' if p_chi < 0.05 else 'is NOT'} significantly associated with pass/fail (p={p_chi:.4f})
3. ANOVA     → Mother's education {'DOES' if p_anova < 0.05 else 'does NOT'} significantly affect grades (p={p_anova:.4f})
4. Bayes     → P(Pass | Study >2hrs/week) = {P_pass_given_study*100:.1f}%

Top Predictors of Student Success:
  ✅ G1, G2 (mid-term grades) — strongest correlation with G3
  ✅ Study time — more hours = higher pass rate
  ✅ Past failures — strong negative impact
  ✅ Mother's education — higher edu = better outcomes
  ✅ Absences — more absences = lower grades

Author : Shrey Dukare | BSIOTR, Pune
Email  : shreydokre@gmail.com
""")
print("✅ Project Complete!")
