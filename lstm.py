import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out)  # Apply FC to all timesteps
        return out  # Shape: [batch_size, seq_length, output_size]

# Load data
df = pd.read_csv('marco_data.csv', sep=';', decimal=',', thousands='.', parse_dates=[1], dayfirst=True)
df = df.dropna(subset=['AE_kWh'])
dataset = df['AE_kWh'].values.reshape(-1, 1)
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

# Generate sequences
def generate_data_windows(dataset, seq_length=10):
    X, Y = [], []
    for i in range(0, len(dataset) - 2*seq_length + 1, seq_length):  # Ensures Y has full length
        print(i, i+seq_length)
        print(i+seq_length, i+2*seq_length)
        X.append(dataset[i:i+seq_length])
        Y.append(dataset[i+seq_length: i+2*seq_length])
    
    return np.array(X), np.array(Y)

seq_length = 10
X_train, Y_train = generate_data_windows(train_dataset, seq_length)
X_test, Y_test = generate_data_windows(test_dataset, seq_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# Define model parameters
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
learning_rate = 0.001
epochs = 100
batch_size = 32

# Initialize model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
predictions = model(X_test_tensor).detach().numpy()
Y_test_np = Y_test_tensor.numpy()

mse_overall = np.mean((predictions - Y_test_np) ** 2)
std_overall = np.std((predictions - Y_test_np) ** 2)

# Save results
with open("lstm_model_results.txt", "w") as f:
    f.write(f"Overall MSE: {mse_overall:.4f} +/- {std_overall:.4f}\n")

# Plot results
for i in range(10):
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test_np[i], label='True')
    plt.plot(predictions[i], label='Predicted')
    plt.legend()
    plt.title("LSTM True vs Predicted")
    plt.savefig(f"lstm_test_{i}.png")
