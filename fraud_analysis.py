# ================================================
# CREDIT CARD FRAUD DETECTION ANALYSIS
# Author: Fatima Iqbal
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Display settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.style.use('seaborn-v0_8-whitegrid')

# Load data
df = pd.read_csv('creditcard.csv')

# Basic overview
print("=== DATASET OVERVIEW ===")
print(f"Total Transactions: {len(df):,}")
print(f"Total Features: {df.shape[1]}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nFraud Rate: {df['Class'].mean() * 100:.3f}%")
print(f"\nTransaction Amount Stats:")
print(df['Amount'].describe())
# ================================================
# STEP 2: CLASS IMBALANCE VISUALIZATION
# ================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Count of fraud vs legitimate
counts = df['Class'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=colors, edgecolor='black')
axes[0].set_title('Transaction Count by Class', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# Chart 2: Percentage breakdown
labels = ['Legitimate\n(99.83%)', 'Fraud\n(0.17%)']
sizes = counts.values
axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.3f%%',
            startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Fraud vs Legitimate Proportion', fontsize=14, fontweight='bold')

plt.suptitle('Class Imbalance Overview', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('01_class_imbalance.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== CLASS IMBALANCE ===")
print(f"Legitimate transactions: {counts[0]:,} ({counts[0]/len(df)*100:.2f}%)")
print(f"Fraudulent transactions: {counts[1]:,} ({counts[1]/len(df)*100:.3f}%)")
print("Chart saved: 01_class_imbalance.png")
# ================================================
# STEP 3: TRANSACTION AMOUNT ANALYSIS
# ================================================

fraud = df[df['Class'] == 1]['Amount']
legit = df[df['Class'] == 0]['Amount']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Chart 1: Amount distribution - Legitimate
axes[0].hist(legit, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[0].set_title('Legitimate Transaction Amounts', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, 1000)

# Chart 2: Amount distribution - Fraud
axes[1].hist(fraud, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1].set_title('Fraudulent Transaction Amounts', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Amount ($)')
axes[1].set_ylabel('Frequency')

# Chart 3: Boxplot comparison
axes[2].boxplot([legit, fraud], labels=['Legitimate', 'Fraud'],
                patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.6))
axes[2].set_title('Amount Distribution Comparison', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Amount ($)')
axes[2].set_ylim(0, 500)

plt.suptitle('Transaction Amount Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_amount_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== AMOUNT ANALYSIS ===")
print(f"Legitimate - Mean: ${legit.mean():.2f} | Median: ${legit.median():.2f} | Max: ${legit.max():.2f}")
print(f"Fraud      - Mean: ${fraud.mean():.2f} | Median: ${fraud.median():.2f} | Max: ${fraud.max():.2f}")
print(f"\nFraud transactions over $1,000: {(fraud > 1000).sum()}")
print(f"Fraud transactions under $100:  {(fraud < 100).sum()} ({(fraud < 100).sum()/len(fraud)*100:.1f}%)")
print("Chart saved: 02_amount_analysis.png")
# ================================================
# STEP 4: STATISTICAL ANOMALY DETECTION
# ================================================

# Z-score method on Amount
df['amount_zscore'] = np.abs(stats.zscore(df['Amount']))

# IQR method on Amount
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
df['amount_iqr_flag'] = ((df['Amount'] < (Q1 - 1.5 * IQR)) |
                          (df['Amount'] > (Q3 + 1.5 * IQR))).astype(int)

# Flag high risk transactions
df['high_risk'] = ((df['amount_zscore'] > 3) |
                   (df['amount_iqr_flag'] == 1)).astype(int)

# How well do our flags catch actual fraud?
flagged = df[df['high_risk'] == 1]
fraud_caught = flagged[flagged['Class'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Z-score distribution
axes[0].hist(df[df['Class']==0]['amount_zscore'], bins=50,
             alpha=0.6, color='#2ecc71', label='Legitimate')
axes[0].hist(df[df['Class']==1]['amount_zscore'], bins=50,
             alpha=0.8, color='#e74c3c', label='Fraud')
axes[0].axvline(x=3, color='black', linestyle='--', label='Z-score threshold (3)')
axes[0].set_title('Z-Score Distribution by Class', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Z-Score')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].set_xlim(0, 20)

# Chart 2: Flag effectiveness
categories = ['Total Flagged', 'Fraud Caught\nby Flags', 'Fraud Missed\nby Flags']
values = [len(flagged), len(fraud_caught), 492 - len(fraud_caught)]
colors = ['#3498db', '#2ecc71', '#e74c3c']
axes[1].bar(categories, values, color=colors, edgecolor='black')
axes[1].set_title('Anomaly Detection Effectiveness', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Transactions')
for i, v in enumerate(values):
    axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.suptitle('Statistical Anomaly Detection', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== ANOMALY DETECTION ===")
print(f"Total transactions flagged as high-risk: {len(flagged):,}")
print(f"Of those, actual fraud: {len(fraud_caught)} ({len(fraud_caught)/len(flagged)*100:.1f}% precision)")
print(f"Fraud cases caught by flags: {len(fraud_caught)} of 492 ({len(fraud_caught)/492*100:.1f}% recall)")
print(f"Fraud cases missed: {492 - len(fraud_caught)}")
print(f"\nIQR bounds: ${Q1 - 1.5*IQR:.2f} to ${Q3 + 1.5*IQR:.2f}")
print("Chart saved: 03_anomaly_detection.png")
# ================================================
# STEP 5: CORRELATION ANALYSIS
# ================================================

# Top features correlated with fraud
correlations = df.corr()['Class'].drop('Class').sort_values()
top_negative = correlations.head(10)
top_positive = correlations.tail(10)
top_features = pd.concat([top_negative, top_positive])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Top correlated features
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features.values]
axes[0].barh(top_features.index, top_features.values, color=colors, edgecolor='black')
axes[0].axvline(x=0, color='black', linewidth=0.8)
axes[0].set_title('Top Features Correlated with Fraud', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Correlation with Fraud (Class)')

# Chart 2: Fraud vs Legit on top 2 features
top_pos_feat = correlations.tail(1).index[0]
top_neg_feat = correlations.head(1).index[0]

axes[1].scatter(df[df['Class']==0][top_neg_feat],
                df[df['Class']==0][top_pos_feat],
                alpha=0.1, color='#2ecc71', label='Legitimate', s=5)
axes[1].scatter(df[df['Class']==1][top_neg_feat],
                df[df['Class']==1][top_pos_feat],
                alpha=0.6, color='#e74c3c', label='Fraud', s=20)
axes[1].set_title(f'Fraud Separation: {top_neg_feat} vs {top_pos_feat}',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel(top_neg_feat)
axes[1].set_ylabel(top_pos_feat)
axes[1].legend()

plt.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('04_correlation_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== CORRELATION ANALYSIS ===")
print("\nTop 5 features POSITIVELY correlated with fraud:")
print(correlations.tail(5).to_string())
print("\nTop 5 features NEGATIVELY correlated with fraud:")
print(correlations.head(5).to_string())
print("\nChart saved: 04_correlation_analysis.png")
# ================================================
# STEP 6: FRAUD SUMMARY TABLE
# ================================================

# Time analysis - convert seconds to hours
df['hour'] = (df['Time'] / 3600) % 24

fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Fraud by hour of day
fraud_by_hour = fraud_df['hour'].apply(lambda x: int(x)).value_counts().sort_index()
axes[0].bar(fraud_by_hour.index, fraud_by_hour.values, color='#e74c3c', edgecolor='black')
axes[0].set_title('Fraudulent Transactions by Hour', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Number of Fraud Cases')

# Chart 2: Summary findings table
summary_data = {
    'Metric': [
        'Total Transactions',
        'Fraudulent Transactions',
        'Fraud Rate',
        'Avg Fraud Amount',
        'Median Fraud Amount',
        'Fraud Under $100',
        'Amount-Only Detection Rate',
        'Top Fraud Indicator'
    ],
    'Finding': [
        '284,807',
        '492',
        '0.173%',
        '$122.21',
        '$9.25',
        '73.6%',
        '18.5%',
        'V11 (r=0.155)'
    ]
}

summary_df = pd.DataFrame(summary_data)
axes[1].axis('off')
table = axes[1].table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc='left',
    loc='center',
    colWidths=[0.55, 0.45]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#ecf0f1')

axes[1].set_title('Key Findings Summary', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Fraud Detection - Final Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('05_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== FINAL SUMMARY ===")
print(summary_df.to_string(index=False))
print("\nChart saved: 05_summary.png")
print("\n=== ALL CHARTS COMPLETE ===")
print("Files saved:")
print("  01_class_imbalance.png")
print("  02_amount_analysis.png")
print("  03_anomaly_detection.png")
print("  04_correlation_analysis.png")
print("  05_summary.png")
# ================================================
# STEP 7: BUSINESS RECOMMENDATIONS (WRITTEN OUTPUT)
# ================================================

recommendations = """
================================================================
CREDIT CARD FRAUD DETECTION - BUSINESS RECOMMENDATIONS
Analyst: Fatima Iqbal
Dataset: 284,807 European cardholder transactions
================================================================

EXECUTIVE SUMMARY
-----------------
Analysis of 284,807 transactions identified 492 confirmed fraud
cases (0.173% fraud rate). Key finding: fraudsters deliberately
avoid triggering amount-based alerts, making behavioral pattern
detection essential for effective fraud prevention.

KEY FINDINGS
------------
1. SMALL TRANSACTION FRAUD IS THE DOMINANT PATTERN
   73.6% of fraudulent transactions were under $100.
   Median fraud amount was $9.25 vs $22.00 for legitimate
   transactions. This indicates systematic card testing behavior
   where stolen credentials are validated with micro-purchases
   before larger attempts.

2. AMOUNT-BASED RULES ALONE ARE INSUFFICIENT
   Statistical flagging based on transaction amount caught only
   18.5% of fraud cases (91 of 492). Rules that flag unusual
   amounts will miss 4 out of 5 fraud cases entirely.

3. BEHAVIORAL FEATURES ARE THE STRONGEST SIGNALS
   Features V11 (r=0.155) and V4 (r=0.133) showed the strongest
   positive correlation with fraud. Features V17 (r=-0.326) and
   V14 (r=-0.303) showed the strongest negative correlation.
   These behavioral patterns are far more predictive than
   transaction amount alone.

4. HIGH FALSE POSITIVE RISK IN CURRENT RULE-BASED SYSTEMS
   Amount-based flagging generated 31,904 alerts but only 0.3%
   were actual fraud. This precision rate would overwhelm fraud
   review teams and create poor customer experience through
   unnecessary transaction blocks.

RECOMMENDATIONS
---------------
1. RETIRE AMOUNT-ONLY FRAUD RULES
   Replace or supplement any existing rules that flag solely
   based on transaction size. The data shows fraudsters have
   adapted to stay below common thresholds.

2. PRIORITIZE BEHAVIORAL FEATURE MONITORING
   Invest in monitoring V11, V4, V17, and V14 as primary fraud
   signals. These features capture transactional behavior that
   is harder for fraudsters to manipulate than simple amounts.

3. IMPLEMENT A TIERED ALERT SYSTEM
   Low risk: Amount anomaly only - monitor passively
   Medium risk: 1-2 behavioral features flagged - soft decline
   High risk: 3+ behavioral features flagged - immediate review

4. BUILD A CARD TESTING DETECTION RULE
   Given 73.6% of fraud occurs under $100, create a specific
   rule that flags accounts with 3+ transactions under $10
   within a 1-hour window as potential card testing activity.

5. REBALANCE TRAINING DATA FOR FUTURE MODELING
   The 0.173% fraud rate creates significant class imbalance.
   Any future machine learning model should use SMOTE or
   weighted sampling to avoid models that simply predict
   legitimate for every transaction.

================================================================
"""

print(recommendations)

# Save to text file
with open('fraud_recommendations.txt', 'w') as f:
    f.write(recommendations)

print("Recommendations saved: fraud_recommendations.txt")
print("\n=== PROJECT COMPLETE ===")
print("Your fraud_detection folder now contains:")
print("  fraud_analysis.py         - Full analysis script")
print("  creditcard.csv            - Dataset")
print("  01_class_imbalance.png    - Chart 1")
print("  02_amount_analysis.png    - Chart 2")
print("  03_anomaly_detection.png  - Chart 3")
print("  04_correlation_analysis.png - Chart 4")
print("  05_summary.png            - Chart 5")
print("  fraud_recommendations.txt - Business recommendations")