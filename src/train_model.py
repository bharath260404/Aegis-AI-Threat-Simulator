import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train():
    # 1. Load Data
    print("Loading processed data...")
    if not os.path.exists("processed_data"):
        print("Error: 'processed_data' folder not found.")
        return

    with open("processed_data/X_train.pkl", "rb") as f: X_train = pickle.load(f)
    with open("processed_data/y_train.pkl", "rb") as f: y_train = pickle.load(f)
    with open("processed_data/X_test.pkl", "rb") as f: X_test = pickle.load(f)
    with open("processed_data/y_test.pkl", "rb") as f: y_test = pickle.load(f)

    # 2. Train XGBoost ( The Heavy Hitter )
    print("Training XGBoost model... (This is faster and often more accurate)")
    # use_label_encoder=False removes a warning
    # eval_metric='logloss' is standard for binary classification
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    
    clf.fit(X_train, y_train)
    print("Training Complete!")

    # 3. Evaluate
    print("\nEvaluating model on Test Data...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 4. Save
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open("models/detector_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("\nXGBoost Model saved to 'models/detector_model.pkl'")

if __name__ == "__main__":
    train()