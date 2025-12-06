# Walmart Retail Sales Analysis

This repository presents a structured exploratory and regression-based assessment of Walmart’s historical weekly sales dataset from 2010-2012. The project evaluates sales patterns, investigates key predictors, and examines the suitability of linear regression models for retail forecasting.



## Contents
- Data importation, cleaning, and preprocessing  
- Exploratory Data Analysis (EDA)  
- Correlation matrix and heatmap  
- Multiple linear regression modelling  
- Regression diagnostics and assumption testing  
- Visualizations: sales distribution, time-series trend, residual behavior  



## Key Insights
- Core predictors (Temperature, Fuel Price, CPI, Unemployment, Holiday Flag) show **weak linear correlation** with Weekly Sales.  
- Diagnostic tests indicate **violations of linearity, homoscedasticity, and normality**, limiting the effectiveness of standard OLS regression.  
- Weekly sales patterns exhibit **non-linear and seasonal behavior**, making them better suited for models such as SARIMA, Prophet, or tree-based regressors (e.g., Random Forest, XGBoost).  



## Tools Used
- Python 3.15
- Libraries: Pandas, NumPy, Statsmodels, Matplotlib, Seaborn  
- Environment: VS Code  
- Version Control: Git & GitHub
- Data source: Walmart Dataset (2010-2012)



## Author
Christopher 
This project represents my first practical and experimental implementation of Python for retail analytics and predictive modelling.
