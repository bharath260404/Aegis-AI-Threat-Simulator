import pandas as pd
import numpy as np
import pickle
import os
from data_loader import load_data

def preprocess_data():
    # 1. Load Data
    train_df, test_df = load_data()

    # 2. Simplify Target (0=Normal, 1=Attack)
    def convert_label(label):
        return 0 if label == 'normal' else 1

    print("Encoding labels...")
    train_df['label'] = train_df['label'].apply(convert_label)
    test_df['label'] = test_df['label'].apply(convert_label)

    # 3. Drop Metadata
    train_df = train_df.drop(['difficulty_level'], axis=1, errors='ignore')
    test_df = test_df.drop(['difficulty_level'], axis=1, errors='ignore')

    # 4. ONE-HOT ENCODING
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    combined = pd.concat([train_df, test_df])

    # Select categorical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    
    print("Applying One-Hot Encoding...")
    combined = pd.get_dummies(combined, columns=cat_cols)

    # Split back
    train_df = combined[combined['source'] == 'train'].drop('source', axis=1)
    test_df = combined[combined['source'] == 'test'].drop('source', axis=1)

    # 5. Separate X and y
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    # --- THE FIX: Force everything to standard Floats ---
    print("Sanitizing data types (Fixing StringDtype error)...")
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    # ----------------------------------------------------

    # 6. Save
    print(f"New Feature Count: {X_train.shape[1]} columns")
    
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")

    print("Saving processed data...")
    with open("processed_data/X_train.pkl", "wb") as f: pickle.dump(X_train, f)
    with open("processed_data/y_train.pkl", "wb") as f: pickle.dump(y_train, f)
    with open("processed_data/X_test.pkl", "wb") as f: pickle.dump(X_test, f)
    with open("processed_data/y_test.pkl", "wb") as f: pickle.dump(y_test, f)

    print("Preprocessing Complete!")

if __name__ == "__main__":
    preprocess_data()