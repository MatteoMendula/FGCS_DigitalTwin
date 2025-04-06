import pandas as pd
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from base_smart_reservoir_search import BaseEpsilonGreedyReservoirHPSearch

class EpsilonGreedyESNSearch(BaseEpsilonGreedyReservoirHPSearch):
    def evaluate(self, params):
        from reservoirpy.nodes import Reservoir, Ridge
        
        # Extract hyperparameters
        units = int(params["units"])
        sr = params["sr"]
        lr = params["learning_rate"]
        alpha = params["ridge"]  # Ridge regression regularization
        activation = params["activation"]
        
        # Define ESN layers
        reservoir1 = Reservoir(units=units, sr=sr, lr=lr, activation=activation)
        reservoir2 = Reservoir(units=units//2, sr=sr, lr=lr, activation=activation)
        readout = Ridge(alpha=alpha)

        # Train the ESN
        states1 = reservoir1.run(self.X_train)
        states2 = reservoir2.run(np.hstack((self.X_train, states1)))
        readout.fit(states2, self.y_train)

        # Test the ESN
        test_states1 = reservoir1.run(self.X_test)
        test_states2 = reservoir2.run(np.hstack((self.X_test, test_states1)))
        predictions = readout.run(test_states2)

        # Compute Mean Squared Error (MSE)
        mse = np.mean((predictions - self.y_test) ** 2)
        return -mse  # Negative MSE (since we maximize score)

if __name__ == '__main__':

    # Load and preprocess data
    df = pd.read_csv('marco_data.csv', sep=';', decimal=',', thousands='.', parse_dates=[1], dayfirst=True)
    df = df.dropna(subset=['AE_kWh'])
    dataset = df['AE_kWh'].values.reshape(-1, 1)  # Ensure proper shape

    train_size = int(len(dataset) * 0.8)
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    # Generate data windows
    def generate_data_windows(dataset, seq_length=10):
        X, Y = [], []
        for i in range(0, len(dataset) - seq_length, seq_length):
            X.append(dataset[i:i+seq_length])
            Y.append(dataset[i+1:i+seq_length+1])
        return np.array(X), np.array(Y)

    seq_length = 50
    X_train, Y_train = generate_data_windows(train_dataset, seq_length)
    X_test, Y_test = generate_data_windows(test_dataset, seq_length)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])  # Ensure shape (samples, features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])      # Same for test set
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])

    # Define deep Echo State Network architecture
    reservoir1 = Reservoir(units=50, sr=0.9, lr=0.1)  # First reservoir
    reservoir2 = Reservoir(units=50, sr=0.9, lr=0.1)  # Second reservoir
    readout = Ridge(ridge=1e-6)  # Readout layer

    # Train the ESN
    states1 = reservoir1.run(X_train)
    states2 = reservoir2.run(np.hstack((X_train, states1)))  # Concatenate input and first reservoir output

    readout.fit(states2, Y_train)

    # Test the ESN
    test_states1 = reservoir1.run(X_test)
    test_states2 = reservoir2.run(np.hstack((X_test, test_states1)))
    predictions = readout.run(test_states2)
    predictions = np.maximum(predictions, 0)

    # Compute error
    mse = np.mean((predictions - Y_test) ** 2)
    print(f'Test MSE: {mse}')

    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(predictions.flatten(), label='Predicted')
    plt.plot(Y_test.flatten(), label='True')
    plt.legend()
    plt.show()