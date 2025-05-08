import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# function for random generation of data
def generate_data(num_samples):
    X = np.random.rand(num_samples, 8)  # create an array filled with random values
    Y = np.zeros((num_samples, 5))  # create a zero array to store y values
    Y[:, 0] = 2 * X[:, 0] * X[:, 2] - X[:, 0] * X[:, 4] + X[:, 2] * X[:, 7] + 2 * X[:, 0]**2 * X[:, 7] + X[:, 4] 
    Y[:, 1] = X[:, 0] * X[:, 4] * X[:, 5] - X[:, 2] * X[:, 3] - 3 * X[:, 1] * X[:, 2] + 2 * X[:, 1]**2 * X[:, 3] - 2 * X[:, 6] * X[:, 7] - 1
    Y[:, 2] = 2 * X[:, 2]**2 - X[:, 4] * X[:, 6] - 3 * X[:, 0] * X[:, 3] * X[:, 5] + X[:, 0]**2 * X[:, 1] * X[:, 3] - 1
    Y[:, 3] = -X[:, 5]**3 + 2.1 * X[:, 0] * X[:, 2] * X[:, 7] - X[:, 0] * X[:, 3] * X[:, 6] - 3.2 * X[:, 4]**2 * X[:, 1] * X[:, 3] - X[:, 7]
    Y[:, 4] = X[:, 0]**2 * X[:, 4] - 3 * X[:, 2] * X[:, 3] * X[:, 7] + X[:, 0] * X[:, 1] * X[:, 3] - 3 * X[:, 5] + X[:, 0]**2 * X[:, 6] + 2
    return X, Y

# function for building the model
def build_model(activations, nodes_per_layer):
    model = Sequential()
    #layer 1
    model.add(Dense(nodes_per_layer[0], activation=activations[0], input_shape=(8,)))
    #layer 2
    model.add(Dense(nodes_per_layer[1], activation=activations[1]))
    #layer 3
    model.add(Dense(nodes_per_layer[2], activation=activations[2]))
    #output layer
    model.add(Dense(5, activation='linear')) 
    return model

# function for model training
def train_model(model, X_train, Y_train, X_test, Y_test, epochs, lr):
    optimizer = SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=1) 
    return history

# generating training and test data
Nt = 1000
Nv = 500
X_train, Y_train = generate_data(Nt)
Y_train_noisy = Y_train + np.random.normal(0, 0.001, size=Y_train.shape)
X_test, Y_test = generate_data(Nv)

# training
initial_nodes_per_layer = [6, 6, 6]
activations = ["relu", "relu", "relu"]  #combination of activation functions
lr = 0.01  # initial learning rate
epochs = 100  # initial number of epochs

model = build_model(activations, initial_nodes_per_layer)
history1 = train_model(model, X_train, Y_train_noisy, X_test, Y_test, epochs, lr)

## trying different configurations
best_activation = None
best_lr = 0.01
best_epochs = 100
best_test_loss = float('inf')

# activation combinations to test
activation_combinations = [
    ["relu", "relu", "relu"],
    ["relu", "sigmoid", "tanh"],
    ["sigmoid", "relu", "tanh"]
]

# learning rates to test
learning_rates = [0.001, 0.01, 0.05]

# number of epochs to test
epochs_list = [100, 150, 175]

results2 = []

# loop through activation combinations, learning rates, and epochs to find the best combination
for activations in activation_combinations:
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"testing with activation: {activations}, learning rate: {lr}, epochs: {epochs}")
            
            model = build_model(activations, initial_nodes_per_layer)
            history2 = train_model(model, X_train, Y_train_noisy, X_test, Y_test, epochs, lr)
            test_loss = history2.history["val_loss"][-1]
            train_loss = history2.history["loss"][-1]
            
            results2.append({
                "activations": activations,
                "learning_rate": lr,
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": test_loss
            })
            
            # checking if combination gives a better test and training loss
            if test_loss < best_test_loss and train_loss < test_loss:
                best_activation = activations
                best_lr = lr
                best_epochs = epochs
                best_test_loss = test_loss

# printing best parameters
print(f"best activation function combination: {best_activation}")
print(f"best learning rate: {best_lr}")
print(f"best number of epochs: {best_epochs}")

## training the model with optimal parameters
# changing number of nodes per layer
nodes_patterns = [[6, 6, 6], [8, 6, 6], [8, 8, 6], [8, 8, 8], [10, 8, 8], [10, 10, 8], [10, 10, 10]]
results5 = []
for nodes_per_layer in nodes_patterns:
    model = build_model(best_activation, nodes_per_layer)
    history5 = train_model(model, X_train, Y_train_noisy, X_test, Y_test, best_epochs, best_lr)
    results5.append({"epochs": epochs, "train_loss": history5.history["loss"][-1], "val_loss": history5.history["val_loss"][-1],
                     "nodes_per_layer": nodes_per_layer})

# plotting bias-variance curve
train_losses = [result["train_loss"] for result in results5]
test_losses = [result["val_loss"] for result in results5]
model_complexities = [sum(result["nodes_per_layer"]) for result in results5]

plt.plot(model_complexities, train_losses, label= "train_loss")
plt.plot(model_complexities, test_losses, label= "val_loss")
plt.xlabel("model complexity (total nodes)")
plt.ylabel("loss")
plt.title("bias-variance curve")
plt.legend()
plt.show()

# increasing training data by 10% and building the model again
Nt_inc = int(Nt * 1.1)
X_train_inc, Y_train_inc = generate_data(Nt_inc)
Y_train_inc_noisy = Y_train_inc + np.random.normal(0, 0.001, size=Y_train_inc.shape)
results6 = []
for nodes_per_layer in nodes_patterns:
    model = build_model(activations, nodes_per_layer)
    history6 = train_model(model, X_train, Y_train_noisy, X_test, Y_test, epochs, lr)
    results6.append({"epochs": epochs, "train_loss": history6.history["loss"][-1], "val_loss": history6.history["val_loss"][-1],
                     "nodes_per_layer": nodes_per_layer})

# plotting bias-variance curve for increased training data
train_losses = [result["train_loss"] for result in results6]
test_losses = [result["val_loss"] for result in results6]
model_complexities = [sum(result["nodes_per_layer"]) for result in results6]

plt.plot(model_complexities, train_losses, label= "train_loss")
plt.plot(model_complexities, test_losses, label= "val_loss")
plt.xlabel("model complexity (total nodes)")
plt.ylabel("loss")
plt.title("bias-variance curve")
plt.legend()
plt.show()
