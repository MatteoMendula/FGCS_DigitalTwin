import pandas as pd
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from base_smart_reservoir_search import BaseEpsilonGreedyReservoirHPSearch

class EpsilonGreedyESNSearch(BaseEpsilonGreedyReservoirHPSearch):
    def evaluate(self, params):

        # Extract hyperparameters
        units = int(params["units"])
        sr = params["sr"]
        lr = params["learning_rate"]
        alpha = params["ridge"]  # Ridge regression regularization
        activation = params["activation"]
        input_scaling = params["input_scaling"]
        seed = params["seed"]
        n_instances = params["n_instances"]
        epochs = params["epochs"]
        warmup = params["warmup"]
        
        # Define ESN layers
        reservoir1 = Reservoir(units=units, sr=sr, lr=lr, activation=activation, input_scaling=input_scaling, seed=seed)
        reservoir2 = Reservoir(units=units//2, sr=sr, lr=lr, activation=activation, input_scaling=input_scaling, seed=seed)
        readout = Ridge(ridge=alpha)

        # Train the ESN
        states1 = reservoir1.run(self.X_train)
        states2 = reservoir2.run(np.hstack((self.X_train, states1)))
        readout.fit(states2, self.y_train, warmup=warmup)

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

    seq_length = 10
    X_train, Y_train = generate_data_windows(train_dataset, seq_length)
    X_test, Y_test = generate_data_windows(test_dataset, seq_length)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])  # Ensure shape (samples, features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])      # Same for test set
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])

    # Initialize search with train and test sets
    hp_search = EpsilonGreedyESNSearch(X_train, Y_train, X_test, Y_test, n_iterations=20, epsilon_greedy=0.3)

    # # Start searching for best hyperparameters
    best_params, best_score = hp_search.search()

    # print("\nBest Hyperparameters Found:", best_params)
    # print("Best Test Score (MSE):", -best_score)

    # best_params = {'units': 500, 'sr': 0.3852529361763695, 'mu': 1.0, 'input_scaling': 0.40036370642322167, 'learning_rate': 2.677709624633953e-05, 'connectivity': 0.23099989346074634, 'activation': 'tanh', 'ridge': 4.696769802550449, 'epochs': 100, 'warmup': 100, 'seed': 12345, 'n_instances': 5, 'score': -0.02573376908735296}

    # plot the best model
    units = int(best_params["units"])
    sr = best_params["sr"]
    lr = best_params["learning_rate"]
    alpha = best_params["ridge"]  # Ridge regression regularization
    activation = best_params["activation"]
    input_scaling = best_params["input_scaling"]
    seed = best_params["seed"]
    n_instances = best_params["n_instances"]
    epochs = best_params["epochs"]
    warmup = best_params["warmup"]

    # Define ESN layers
    reservoir1 = Reservoir(units=units, sr=sr, lr=lr, activation=activation, input_scaling=input_scaling, seed=seed)
    reservoir2 = Reservoir(units=units//2, sr=sr, lr=lr, activation=activation, input_scaling=input_scaling, seed=seed)
    readout = Ridge(ridge=alpha)

    # Train the ESN
    states1 = reservoir1.run(X_train)
    states2 = reservoir2.run(np.hstack((X_train, states1)))
    readout.fit(states2, Y_train, warmup=warmup)

    # Test the ESN
    test_states1 = reservoir1.run(X_test)
    test_states2 = reservoir2.run(np.hstack((X_test, test_states1)))

    # Predictions
    predictions = readout.run(test_states2)
    # remove dimensions equal to 1
    # predictions = np.squeeze(predictions)
    print(predictions.shape)
    print(Y_test.shape)

    # calc overall mse
    mse_overall = np.mean((predictions - Y_test) ** 2)
    std_overall = np.std((predictions - Y_test) ** 2)

    # calc mse for the last index of each prediction
    mse_last = np.mean((predictions[:, -1] - Y_test[:, -1]) ** 2)
    std_last = np.std((predictions[:, -1] - Y_test[:, -1]) ** 2)

    # save on file
    with open("model_results.txt", "w") as f:
        f.write(f"Overall MSE: {mse_overall:.4f} +/- {std_overall:.4f}\n")
        f.write(f"Last index MSE: {mse_last:.4f} +/- {std_last:.4f}\n")

    
    # Plot the predictions
    import matplotlib.pyplot as plt
    for i in range(len(Y_test)):
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test[i], label='True')
        plt.plot(predictions[i], label='Predicted')
        plt.legend()
        plt.title("True vs Predicted")
        plt.show()


