# ===============[ IMPORTS ]===============
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ===============[ LOADING THE DATA ]===============
data = pd.read_csv('data.csv')
data.test = [1 if diagnosis == 'positif' else 0 for diagnosis in data.test]


# ===============[ INITIALIZING THE DATASET ]===============
Y = np.transpose([data.test])
data.drop('test', axis=1, inplace=True)
X = data.values
X = (X - np.min(X)) / np.max(X) - np.min(X)
X, X_evaluate, Y, Y_evaluate = train_test_split(X, Y, test_size=0.15, random_state=42, shuffle=True)

m = X.shape[0]
n = X.shape[1]

w = np.random.randn(n, 1)
b = np.random.randn()


# ===============[ HELPER FONCTIONS ]===============
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ===============[ FORWARD PROPAGATION ]===============
def forward_propagate(X, w, b):
    z = (X @ w) + b
    return sigmoid(z)


# ===============[ FORWARD / BACKWARD PROPAGATION ]===============
def forward_backward_propagation(weights, bias):
    activations = sigmoid((X @ weights) + bias)
    m = activations.shape[0]
    cost = np.sum(-Y * np.log(activations) - ((1 - Y) * np.log(1 - activations))) / activations.shape[0]

    gradient_weights = (X.T @ (activations - Y)) / m
    gradient_bias = np.sum(activations - Y) / m
    gradients = {'weights': gradient_weights, 'bias': gradient_bias}
    return activations, cost, gradients


# ===============[ EVALUATING ACCURACY ]===============
def compute_efficiency(X, Y, parameters):
    activations = forward_propagate(X, parameters['weights'], parameters['bias'])
    successes = 0
    m = X.shape[0]

    for i in range(m):
        if (activations[i, 0] >= 0.5 and Y[i, 0] == 1) or (activations[i, 0] < 0.5 and Y[i, 0] == 0):
            successes = successes + 1

    return (successes / m) * 100


# ===============[ EVALUATING COST ]===============
def compute_cost(activations):
    return np.sum(-Y * np.log(activations) - ((1 - Y) * np.log(1 - activations))) / m


def plot_cost(costs, indexes):
    plt.plot(indexes, costs)
    plt.xticks(indexes, rotation='vertical')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


# ===============[ COMPUTING GRADIENTS ]===============
def compute_gradients(activations):
    gradient_weights = (X.T @ (activations - Y)) / m
    gradient_bias = np.sum(activations - Y) / m
    return {'weights': gradient_weights, 'bias': gradient_bias}


# ===============[ OPTIMIZING USING GRADIENT DESCENT ]===============
def gradient_descent(w, b, alpha, iterations):
    cost_list = []
    index_list = []

    for i in range(iterations + 1):
        activations = forward_propagate(X, w, b)
        cost = compute_cost(activations)
        gradients = compute_gradients(activations)

        w = w - alpha * gradients['weights']
        b = b - alpha * gradients['bias']

        if i % 10 == 0:
            cost_list.append(cost)
            index_list.append(i)

    parameters = {'weights': w, 'bias': b}
    return parameters, cost_list, index_list


# ===============[ TRAINING THE CLASSIFIER ]===============
parameters, cost_list, index_list = gradient_descent(w, b, alpha=5, iterations=1000)
plot_cost(cost_list, index_list)

training_efficiency = compute_efficiency(X, Y, parameters)
evaluation_efficiency = compute_efficiency(X_evaluate, Y_evaluate, parameters)

print('EFFICIENCY (Training): \t{}% '.format(training_efficiency))
print('EFFICIENCY (Testing): \t{}% '.format(evaluation_efficiency))
