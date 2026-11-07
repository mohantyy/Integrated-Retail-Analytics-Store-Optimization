"""
Integrated Retail Analytics for Store Optimization and Demand Forecasting
Author: Abhilash Mohanty (@mohantyy)
"""
import numpy as np
import pandas as pd

class RetailDataPreprocessor:
    MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

    def clean_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["Date_Parsed"] = pd.to_datetime(data["Date"], format="%d/%m/%Y", errors="coerce")
        data = data.sort_values(["Store", "Dept", "Date_Parsed"]).reset_index(drop=True)
        for col in self.MARKDOWN_COLS:
            if col in data.columns:
                data[col] = np.maximum(0.0, pd.to_numeric(data[col], errors="coerce").fillna(0.0))
        data["Total_MarkDown"] = data[self.MARKDOWN_COLS].sum(axis=1)
        data["Has_Promo_Markdown"] = (data["Total_MarkDown"] > 0).astype(int)
        for macro_col in ["CPI", "Unemployment", "Temperature", "Fuel_Price"]:
            if macro_col in data.columns:
                data[macro_col] = data.groupby("Store")[macro_col].transform(
                    lambda s: s.interpolate(method="linear").bfill().ffill()
                )
        data["Is_Return_Adjustment"] = (data["Weekly_Sales"] < 0).astype(int)
        data["Weekly_Sales_Clean"] = np.maximum(0.0, data["Weekly_Sales"])
        data["Week"] = data["Date_Parsed"].dt.isocalendar().week.astype(int)
        data["Week_Sin"] = np.sin(2.0 * np.pi * data["Week"] / 52.0)
        data["Week_Cos"] = np.cos(2.0 * np.pi * data["Week"] / 52.0)
        data["Sales_Per_SqFt"] = data["Weekly_Sales_Clean"] / np.maximum(1.0, data["Size"])
        grp = data.groupby(["Store", "Dept"])["Weekly_Sales_Clean"]
        data["Lag_1"] = grp.shift(1).bfill()
        data["Lag_4"] = grp.shift(4).bfill()
        data["Rolling_Mean_4w"] = grp.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()).bfill()
        return data
