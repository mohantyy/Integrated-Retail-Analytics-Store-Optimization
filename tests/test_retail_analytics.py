"""Unit Test Suite for Integrated Retail Analytics Pipeline."""
import unittest
import pandas as pd
from src.retail_analytics_pipeline import RetailDataPreprocessor

class TestIntegratedRetailAnalytics(unittest.TestCase):
    def test_preprocessing_and_imputation(self):
        sample = pd.DataFrame({
            "Store": [1, 1],
            "Dept": [1, 1],
            "Date": ["05/02/2010", "12/02/2010"],
            "Weekly_Sales": [24924.50, -45.00],
            "IsHoliday": [False, True],
            "Temperature": [42.31, 38.51],
            "Fuel_Price": [2.572, 2.548],
            "MarkDown1": [None, 1250.0],
            "MarkDown2": [None, 0.0],
            "MarkDown3": [None, 450.0],
            "MarkDown4": [None, 0.0],
            "MarkDown5": [None, 0.0],
            "CPI": [211.096, None],
            "Unemployment": [8.106, 8.106],
            "Type": ["A", "A"],
            "Size": [151315, 151315],
        })
        clean_df = RetailDataPreprocessor().clean_and_engineer(sample)
        self.assertEqual(clean_df["MarkDown1"].isna().sum(), 0)
        self.assertEqual(clean_df["CPI"].isna().sum(), 0)
        self.assertTrue((clean_df["Weekly_Sales_Clean"] >= 0).all())
        self.assertIn("Sales_Per_SqFt", clean_df.columns)

if __name__ == "__main__":
    unittest.main()
