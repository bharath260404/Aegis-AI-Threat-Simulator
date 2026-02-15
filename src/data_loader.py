import pandas as pd
import os
import urllib.request

# Define column names
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty_level"]

# Direct Raw Links to the Dataset (Hosted on GitHub)
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

def load_data():
    # Ensure datasets folder exists
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        print("Created 'datasets' folder.")

    train_path = os.path.join("datasets", "KDDTrain+.txt")
    test_path = os.path.join("datasets", "KDDTest+.txt")

    # Download Train Data if missing
    if not os.path.exists(train_path):
        print("Downloading Training Data... (This might take a minute)")
        urllib.request.urlretrieve(TRAIN_URL, train_path)
        print("Training Data Downloaded!")

    # Download Test Data if missing
    if not os.path.exists(test_path):
        print("Downloading Test Data...")
        urllib.request.urlretrieve(TEST_URL, test_path)
        print("Test Data Downloaded!")

    print("\nLoading data into Pandas...")
    
    # Read CSV
    train_df = pd.read_csv(train_path, names=col_names)
    test_df = pd.read_csv(test_path, names=col_names)

    print(f"Training Data Loaded: {train_df.shape}")
    print(f"Testing Data Loaded: {test_df.shape}")
    
    # Show the first few rows
    print("\nSample Data:")
    print(train_df[['duration', 'service', 'flag', 'label']].head())
    
    return train_df, test_df

if __name__ == "__main__":
    load_data()