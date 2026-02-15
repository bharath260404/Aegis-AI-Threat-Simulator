import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os

# Import the architecture we defined earlier
from gan_model import Generator, Discriminator

# Configuration
BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 50  # Number of times to loop through the data
Z_DIM = 100   # Size of the random noise vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gan():
    print("--- 🔴 Phase 2: Initializing Red Team (GAN Training) ---")
    
    # 1. Load Data
    print("Loading processed data...")
    with open("processed_data/X_train.pkl", "rb") as f: X_train = pickle.load(f)
    with open("processed_data/y_train.pkl", "rb") as f: y_train = pickle.load(f)

    # 2. Filter: We only want to learn from ATTACKS (Label = 1)
    # We want the Generator to become a master hacker, so we feed it only attack data.
    attack_indices = np.where(y_train == 1)[0]
    X_attacks = X_train.iloc[attack_indices].values
    
    # Convert to PyTorch Tensors
    tensor_x = torch.Tensor(X_attacks).to(device)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training on {len(X_attacks)} attack samples.")

    # 3. Initialize Models
    input_dim = X_train.shape[1]
    generator = Generator(Z_DIM, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # Optimizers (The "Brain" that updates the weights)
    optimizer_G = optim.Adam(generator.parameters(), lr=LR)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

    # Loss Function
    criterion = nn.BCELoss()

    # 4. The Training Loop (The War)
    print("Starting Adversarial Training...")
    
    for epoch in range(EPOCHS):
        for i, (real_attacks,) in enumerate(dataloader):
            
            # --- A. Train Discriminator (The Police) ---
            # Goal: Correctly label Real attacks as "1" and Fake attacks as "0"
            optimizer_D.zero_grad()
            
            # Real Data
            real_labels = torch.ones(real_attacks.size(0), 1).to(device)
            output_real = discriminator(real_attacks)
            loss_real = criterion(output_real, real_labels)

            # Fake Data (Generator creates attacks from noise)
            z = torch.randn(real_attacks.size(0), Z_DIM).to(device)
            fake_attacks = generator(z)
            fake_labels = torch.zeros(real_attacks.size(0), 1).to(device)
            output_fake = discriminator(fake_attacks.detach()) # Detach so G doesn't learn yet
            loss_fake = criterion(output_fake, fake_labels)

            # Combine Loss & Update Police
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # --- B. Train Generator (The Hacker) ---
            # Goal: Fool the Police (Make Discriminator label Fakes as "1")
            optimizer_G.zero_grad()
            
            # We want the discriminator to think these fakes are REAL (1)
            target_labels = torch.ones(real_attacks.size(0), 1).to(device)
            output_fake_for_G = discriminator(fake_attacks)
            loss_G = criterion(output_fake_for_G, target_labels)
            
            # Update Hacker
            loss_G.backward()
            optimizer_G.step()

        # Print stats every 10 epochs
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    # 5. Save the Generator (We only need the attacker now)
    if not os.path.exists("models"):
        os.makedirs("models")
    
    torch.save(generator.state_dict(), "models/generator_model.pth")
    print("\n✅ Generator Trained & Saved to 'models/generator_model.pth'")

if __name__ == "__main__":
    train_gan()