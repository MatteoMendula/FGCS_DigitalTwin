import pandas as pd
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from base_smart_reservoir_search import BaseEpsilonGreedyReservoirHPSearch
import numpy as np
import random
from collections import deque
import json
import matplotlib.pyplot as plt

class BaseEpsilonGreedyReservoirHPSearch:
    def __init__(self, X_train, y_train, X_test, y_test, n_iterations, n_reservoirs, epsilon_greedy=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_reservoirs = n_reservoirs  # Allow a dynamic number of reservoirs
        self.memory = deque(maxlen=5)
        self.history = deque(maxlen=n_iterations)
        self.epsilon_greedy = epsilon_greedy
        self.best_score = -np.inf
        self.best_params = None

        # Define independent search spaces for each reservoir
        self.search_space = {
            f"reservoir_{i}": {
                "units": [50, 500, 5000],  # Discrete choices
                "sr": {"min": np.log(1e-2), "max": np.log(1e1)},
                "mu": [0, 1],  # Linear scale for mu as it's between 0 and 1
                "input_scaling": {"min": np.log(1e-5), "max": np.log(2e2)},
                "learning_rate": {"min": np.log(1e-5), "max": np.log(1e-2)},
                "connectivity": [0.1, 0.5],  # Linear scale for probability
                "activation": ["sigmoid", "tanh"],
                "ridge": {"min": np.log(1e-8), "max": np.log(1e1)},
            }
            for i in range(n_reservoirs)
        }

        # Global parameters
        self.global_params = {
            "seed": 12345,
            "n_instances": 5,
            "epochs": 100,
            "warmup": 100
        }

    def log_uniform_sample(self, param_range):
        return np.exp(np.random.uniform(param_range["min"], param_range["max"]))

    def memory_guided_sample(self):
        """Sample hyperparameters for multiple reservoirs dynamically."""
        if len(self.memory) > 0 and random.random() < (1 - self.epsilon_greedy if self.epsilon_greedy is not None else 0.7):
            base_config = random.choice(list(self.memory))
            params = {}
            
            for res_key, search_params in self.search_space.items():
                params[res_key] = {
                    key: (self.log_uniform_sample(value) if isinstance(value, dict) else random.choice(value))
                    for key, value in search_params.items()
                }
        else:
            params = {
                f"reservoir_{i}": {
                    "units": random.choice(self.search_space[f"reservoir_{i}"]["units"]),
                    "sr": self.log_uniform_sample(self.search_space[f"reservoir_{i}"]["sr"]),
                    "mu": random.uniform(*self.search_space[f"reservoir_{i}"]["mu"]),
                    "input_scaling": self.log_uniform_sample(self.search_space[f"reservoir_{i}"]["input_scaling"]),
                    "learning_rate": self.log_uniform_sample(self.search_space[f"reservoir_{i}"]["learning_rate"]),
                    "connectivity": random.uniform(*self.search_space[f"reservoir_{i}"]["connectivity"]),
                    "activation": random.choice(self.search_space[f"reservoir_{i}"]["activation"]),
                    "ridge": self.log_uniform_sample(self.search_space[f"reservoir_{i}"]["ridge"]),
                }
                for i in range(self.n_reservoirs)
            }

        params.update(self.global_params)
        return params

    def search(self, n_iterations=20):
        for i in range(n_iterations):
            params = self.memory_guided_sample()
            score = self.evaluate(params)
            params["score"] = score
            self.history.append(params)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.memory.append(params)
                print(f"New best score: {score:.4f}")
            
            print(f"Iteration {i+1}/{n_iterations}: Score = {score:.4f}")
        
        return self.best_params, self.best_score

    def evaluate(self, params):
        raise NotImplementedError("Subclasses must implement evaluate method")


import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

class EpsilonGreedyESNSearch(BaseEpsilonGreedyReservoirHPSearch):
    def evaluate(self, params):
        reservoirs = []
        train_state_list = []
        test_state_list = []

        for i in range(self.n_reservoirs):
            res_params = params[f"reservoir_{i}"]
            reservoir = Reservoir(
                units=res_params["units"],
                sr=res_params["sr"],
                lr=res_params["learning_rate"],
                activation=res_params["activation"],
                input_scaling=res_params["input_scaling"],
                seed=params["seed"]
            )
            reservoirs.append(reservoir)

        # Forward pass through reservoirs
        last_train_states = None
        for res in reservoirs:
            if last_train_states is None:
                last_train_states = res.run(self.X_train)
            train_state_list.append(last_train_states)

        current_states = np.hstack(train_state_list)  # Stack new state with input

        # Ridge regression as readout
        readout = Ridge(ridge=params["reservoir_0"]["ridge"])  # Use first reservoir's ridge param
        readout.fit(current_states, self.y_train, warmup=params["warmup"])

        # Test the ESN
        last_test_states = None
        for res in reservoirs:
            if last_test_states is None:
                last_test_states = res.run(self.X_test)
            test_state_list.append(last_test_states)

        test_states = np.hstack(test_state_list)

        predictions = readout.run(test_states)

        # Compute Negative Mean Squared Error (MSE)
        mse = np.mean((predictions - self.y_test) ** 2)

        # Compute Negative Mean Squared Error (MSE) on the last index
        # mse = np.mean((predictions[:, -1] - self.y_test[:, -1]) ** 2)

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
        for i in range(0, len(dataset) - 2*seq_length + 1, seq_length):  # Ensures Y has full length
            print(i, i+seq_length)
            print(i+seq_length, i+2*seq_length)
            X.append(dataset[i:i+seq_length])
            Y.append(dataset[i+seq_length: i+2*seq_length])
        
        return np.array(X), np.array(Y)

    seq_length = 10
    X_train, Y_train = generate_data_windows(train_dataset, seq_length)
    X_test, Y_test = generate_data_windows(test_dataset, seq_length)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    # print(range(0, len(dataset) - seq_length, seq_length)[0])
    # print(range(0, len(dataset) - seq_length, seq_length)[1])
    # # plot first 10 samples train
    # for i in range(10):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(X_train[i], label='Input')
    #     plt.plot(X_train[i+1], label='Input + 1')
    #     # plt.plot(Y_train[i], label='Target')
    #     plt.legend()
    #     plt.title("Input vs Target Energy Consumption")
    #     plt.show()


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])  # Ensure shape (samples, features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])      # Same for test set
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])

    hp_search = EpsilonGreedyESNSearch(X_train, Y_train, X_test, Y_test, n_iterations=20, n_reservoirs=3, epsilon_greedy=0.3)
    best_params, best_score = hp_search.search()

    print(f"Best score: {best_score:.4f}")
    best_params["score"] = best_score
    # save best params
    with open("best_params.json", "w") as f:
        json.dump(best_params, f)

    # plot the best model
    reservoirs = []
    state_list = []

    for i in range(hp_search.n_reservoirs):
        res_params = best_params[f"reservoir_{i}"]
        reservoir = Reservoir(
            units=res_params["units"],
            sr=res_params["sr"],
            lr=res_params["learning_rate"],
            activation=res_params["activation"],
            input_scaling=res_params["input_scaling"],
            seed=best_params["seed"]
        )
        reservoirs.append(reservoir)

    # Forward pass through reservoirs
    current_states = X_train
    for res in reservoirs:
        state = res.run(current_states)
        state_list.append(state)
        current_states = np.hstack((current_states, state))  # Stack new state with input

    # Ridge regression as readout
    readout = Ridge(ridge=best_params["reservoir_0"]["ridge"])  # Use first reservoir's ridge param
    readout.fit(current_states, Y_train, warmup=best_params["warmup"])

    # Test the ESN
    test_states = X_test
    for res in reservoirs:
        test_state = res.run(test_states)
        test_states = np.hstack((test_states, test_state))

    predictions = readout.run(test_states)

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

    for i in range(len(Y_test))[:10]:
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test[i], label='True')
        plt.plot(predictions[i], label='Predicted')
        plt.legend()
        plt.title("True vs Predicted")
        # plt.show()
        plt.savefig(f"test_{i}.png")
