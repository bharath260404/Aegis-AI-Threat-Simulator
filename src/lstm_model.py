import torch
import torch.nn as nn

class CyberLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(CyberLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 1. The LSTM Layer (The Memory)
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        
        # 2. The Fully Connected Layer (The Decision)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 3. Activation (Probability)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden state and cell state (The "Short" and "Long" term memory)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We only care about the LAST time step's output (did the sequence END in an attack?)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

if __name__ == "__main__":
    # Test the architecture
    print("Initializing Military-Grade Neural Network...")
    model = CyberLSTM(input_dim=122, hidden_dim=64, output_dim=1, n_layers=2)
    print(model)
    print("✅ System Ready.")