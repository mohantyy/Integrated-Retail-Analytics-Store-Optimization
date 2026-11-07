# Integrated Retail Analytics for Store Optimization and Demand Forecasting

**Author:** Abhilash Mohanty (@mohantyy)
**Module:** AlmaBetter Masters in Data Science — Advanced Machine Learning Capstone

## Overview & Key Results
End-to-end Machine Learning system across **45 retail stores**, **12 core retail departments**, and **143 historical weeks (Feb 2010 – Oct 2012)** encompassing **77,220 store-department-week observations** merged from `sales data-set.csv`, `Features data set.csv`, and `stores data-set.csv`.

1. **Data Preprocessing & Feature Engineering (`RetailDataPreprocessor`)**: Imputes `MarkDown1–5` pre-Nov 2011 sparsity with `0.0` + `Has_Promo_Markdown` flag (35.66% active weeks), interpolates store-level `CPI` & `Unemployment`, isolates `88` return adjustments, and engineers **39 domain features** (`Sales_Per_SqFt`, `Holiday_Type`, `Week_Sin`, `Week_Cos`, `Lag_1`, `Lag_2`, `Lag_4`, `Rolling_Mean_4w`).
2. **Point & Time-Series Anomaly Detection (`RetailAnomalyDetector`)**: Combines Modified Z-Scores, IQR Fences, 5D Mahalanobis Distance, and STL Seasonal Decomposition residuals (`|Z| > 3.0`) to flag **3,503 consensus anomalies (4.54%)**: Promotional Spikes (`1,320`), Unexplained Shocks (`1,191`), Holiday Surges (`904`), and Return Adjustments (`88`).
3. **Store & Department Segmentation (`StoreDepartmentSegmenter`)**: Evaluates K-Means++ (`Silhouette = 0.4906`, `Davies-Bouldin = 0.7675`, `Calinski-Harabasz = 76.67`) and Ward Hierarchical Clustering (`Silhouette = 0.5404`, `Davies-Bouldin = 0.5978`, `Calinski-Harabasz = 77.55`) across k=3 store personas: Tier-1 Flagship Megastores (15), Tier-2 Suburban Growth Stores (17), and Tier-3 Neighborhood Compact Stores (13).
4. **Market Basket Analysis (`DepartmentMarketBasketAnalyzer`)**: Mines **110 cross-department association rules** (`Support >= 0.14`, `Confidence >= 0.42`), led by **Health & Beauty Aids -> Household Paper Goods** (`Support = 0.2733`, `Confidence = 68.58%`, `Lift = 1.7204`).
5. **Demand Forecasting (`RetailDemandForecaster`)**: Evaluates Ridge Regression, Random Forest Ensemble, and **Champion Gradient Boosted Decision Trees (GBT)** on an 80/20 out-of-time split:

| Model Architecture | WMAE ($) (5x Holiday Weight) | RMSE ($) | MAE ($) | Out-of-Time R2 |
| :--- | :---: | :---: | :---: | :---: |
| Ridge Regression (Baseline) | $6,320.77 | $10,745.00 | $5,854.88 | 0.9177 |
| Ridge Regression (Tuned) | $6,079.14 | $10,557.86 | $5,579.60 | 0.9206 |
| Random Forest Ensemble (Baseline) | $10,383.79 | $17,881.19 | $10,047.82 | 0.7722 |
| Random Forest Ensemble (Tuned) | $7,049.21 | $12,260.60 | $6,617.72 | 0.8929 |
| Gradient Boosted Trees (Baseline) | $6,139.19 | $10,632.25 | $5,716.19 | 0.9195 |
| **Gradient Boosted Trees (Tuned — Champion)** | **$5,900.82** | **$10,514.75** | **$5,512.93** | **0.9212** |
