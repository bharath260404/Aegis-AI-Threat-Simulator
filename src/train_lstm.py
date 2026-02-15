import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import os
from lstm_model import CyberLSTM

# --- CONFIGURATION (The "Expensive" Settings) ---
SEQUENCE_LENGTH = 10  # Look at 10 packets at a time
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_DIM = 128      # Big brain size
LAYERS = 2            # Deep network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    # Sliding window: Move 1 step at a time
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length] # Predict the NEXT label
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm():
    print(f"--- 🧠 Phase 3: Initializing Deep Learning Core (LSTM) on {device} ---")
    
    # 1. Load Data
    print("Loading raw intelligence...")
    with open("processed_data/X_train.pkl", "rb") as f: X_train = pickle.load(f)
    with open("processed_data/y_train.pkl", "rb") as f: y_train = pickle.load(f)
    
    # Convert to Numpy for fast slicing
    X_train = X_train.values
    y_train = y_train.values

    # 2. Reshape for Time-Series (The "Pro" Move)
    print(f"Creating temporal sequences (Window Size: {SEQUENCE_LENGTH})...")
    # We take a subset for speed in this demo, or use full if you have a GPU
    # Using first 20,000 samples for quick training demonstration
    X_seq, y_seq = create_sequences(X_train[:20000], y_train[:20000], SEQUENCE_LENGTH)
    
    # Convert to PyTorch Tensors
    train_data = TensorDataset(torch.from_numpy(X_seq).float(), torch.from_numpy(y_seq).float())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    
    print(f"Training on {len(X_seq)} sequences.")

    # 3. Initialize Model
    input_dim = X_train.shape[1]
    model = CyberLSTM(input_dim, HIDDEN_DIM, 1, LAYERS).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting Deep Neural Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # 5. Save
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("\n✅ LSTM Neural Core Saved to 'models/lstm_model.pth'")

if __name__ == "__main__":
    train_lstm()