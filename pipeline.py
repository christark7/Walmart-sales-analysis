"""Walmart Sales Analysis - Compact Pipeline"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# setup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'data', 'walmart.csv')
output_clean_dir = os.path.join(base_dir, '..', 'output', 'cleaned_data')
charts_dir = os.path.join(base_dir, '..', 'output', 'charts')
os.makedirs(output_clean_dir, exist_ok=True)
os.makedirs(charts_dir, exist_ok=True)

# load and clean
df = pd.read_csv(data_path)
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df = df.sort_values(['store', 'date']).reset_index(drop=True)
df = df.drop_duplicates()

# handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# fix types
df['store'] = df['store'].astype(int)
df['holiday_flag'] = df['holiday_flag'].astype(int)
df['weekly_sales'] = df['weekly_sales'].astype(float)

# feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week

# calculate adjusted weekly sales (normalized by store mean)
store_means = df.groupby('store')['weekly_sales'].transform('mean')
df['adj_weekly_sales'] = df['weekly_sales'] / store_means

# prepare regression data
features = ['temperature', 'fuel_price', 'cpi', 'unemployment', 'holiday_flag']
target = 'adj_weekly_sales'
reg_df = df[features + [target]].dropna()

x = reg_df[features]
y = reg_df[target]
x_const = sm.add_constant(x)

# fit ols model
model = sm.OLS(y, x_const).fit()

# print summary
print("="*80)
print("WALMART SALES OLS REGRESSION SUMMARY")
print("="*80)
print(model.summary())
print("\n" + "="*80)
print("KEY METRICS")
print("="*80)
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.6f})")
print(f"Observations: {int(model.nobs)}")
print("\nCOEFFICIENTS:")
for name, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {name:20s}: {coef:+10.4f}  (p={pval:.4f}) {sig}")

# save cleaned data
clean_path = os.path.join(output_clean_dir, 'walmart_clean.csv')
df.to_csv(clean_path, index=False)
print(f"\n✓ Cleaned data saved: {clean_path}")

# save regression summary (text) and coefficients (csv) for reproducibility
summary_text = model.summary().as_text()
summary_path = os.path.join(output_clean_dir, 'regression_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as fh:
    fh.write(summary_text)
print(f"✓ Regression summary saved: {summary_path}")

coef_df = pd.DataFrame({
    'coef': model.params,
    'std_err': model.bse,
    't_value': model.tvalues,
    'p_value': model.pvalues
})
coef_path = os.path.join(output_clean_dir, 'regression_coefficients.csv')
coef_df.to_csv(coef_path)
print(f"✓ Regression coefficients saved: {coef_path}")

# create coefficient plot
fig, ax = plt.subplots(figsize=(10, 6))
coefs = model.params[1:]  # exclude intercept
errors = model.bse[1:]
indices = np.arange(len(coefs))

colors = ['green' if p < 0.05 else 'gray' for p in model.pvalues[1:]]
ax.barh(indices, coefs, xerr=errors, color=colors, alpha=0.7, capsize=5)
ax.set_yticks(indices)
ax.set_yticklabels(coefs.index)
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('OLS Regression Coefficients\n(Green = Significant at p<0.05)', 
            fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'coefficient_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Coefficient plot saved: ../output/charts/coefficient_plot.png")

# create residual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# residuals vs fitted
axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.3, s=5)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].grid(True, alpha=0.3)

# qq plot
sm.qqplot(model.resid, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')
axes[0, 1].grid(True, alpha=0.3)

# scale-location
std_resid = np.sqrt(np.abs(model.resid_pearson))
axes[1, 0].scatter(model.fittedvalues, std_resid, alpha=0.3, s=5)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].grid(True, alpha=0.3)

# residual histogram
axes[1, 1].hist(model.resid, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residuals Distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Diagnostics plot saved: ../output/charts/diagnostics.png")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)