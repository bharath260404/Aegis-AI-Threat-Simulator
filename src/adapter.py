import pandas as pd
import pickle
import numpy as np
import os

def adapt_file(uploaded_file_path, model_columns_path="processed_data/X_test.pkl"):
    print(f"🔄 Adapting {uploaded_file_path} to match AI Model...")

    # 1. Load the "Template" (The columns your model expects)
    if not os.path.exists(model_columns_path):
        return None, "❌ Error: processed_data/X_test.pkl not found. Train your model first."
        
    with open(model_columns_path, "rb") as f:
        template_df = pickle.load(f)
        expected_cols = template_df.columns.tolist()

    # 2. Load the Real File
    try:
        # Real files often have spaces in column names (e.g., " Destination Port")
        real_df = pd.read_csv(uploaded_file_path)
        real_df.columns = real_df.columns.str.strip() # Clean spaces
    except Exception as e:
        return None, f"❌ Error reading CSV: {e}"

    # 3. THE MAGIC: Align Columns
    # Create a new empty dataframe with the EXACT structure the AI wants
    adapted_df = pd.DataFrame(columns=expected_cols)

    # Fill common columns (e.g., if both have 'Destination Port', copy the data)
    common_cols = list(set(real_df.columns) & set(expected_cols))
    adapted_df[common_cols] = real_df[common_cols]

    # Fill missing columns with 0 (Safe assumption for missing features)
    adapted_df = adapted_df.fillna(0)

    # 4. Limit to 122 columns (Drop extras)
    adapted_df = adapted_df[expected_cols]

    print("✅ Adaptation Complete. File is ready for AI.")
    return adapted_df, None