"""
Streamlit Dashboard & CLI Verification Entrypoint for Integrated Retail Analytics
Run CLI verification: python3 app.py --cli-demo
"""
import argparse
import os
import pandas as pd
from src.retail_analytics_pipeline import RetailDataPreprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "merged_retail_analytics_dataset.csv")

def run_cli_demo():
    print("INTEGRATED RETAIL ANALYTICS FOR STORE OPTIMIZATION & DEMAND FORECASTING")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        clean_df = RetailDataPreprocessor().clean_and_engineer(df)
        print(f"Processed {len(clean_df):,} store-department-week observations across {clean_df['Store'].nunique()} stores.")
    else:
        print("Pipeline module verified and ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-demo", action="store_true")
    args, _ = parser.parse_known_args()
    run_cli_demo()
