# ===============[ IMPORTATION DES MODULES ]===============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============[ CHARGEMENT DE LA DONNÉE ]===============
data = pd.read_csv('data.csv')
data.test = [1 if diagnosis == 'positif' else 0 for diagnosis in data.test]

Y = np.transpose([data.test])
data.drop('test', axis=1, inplace=True)

X = data.values
X = (X - np.min(X)) / np.max(X) - np.min(X)
X = X.T


# ===============[ DÉFINITION DES DIMENSIONS DES COUCHES ]===============
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_cost(costs, indexes):
    plt.plot(indexes, costs)
    plt.xticks(indexes, rotation='vertical')
    plt.xlabel('Itérations')
    plt.ylabel('Coût')
    plt.show()


# ===============[ DÉFINITION DES DIMENSIONS DES COUCHES ]===============
def layer_sizes(X, Y, hidden_units):
    """
    Arguments:
    X -- dataset d'entrée (nombre_exemples, nombre features)
    Y -- étiquettes du dataset (nombre_exemples, nombre sorties)

    Retourne:
    n_x -- taille de la couche d'entrée
    n_h -- taille de la couche cachée
    n_y -- taille de la couche de sortie
    """

    n_x = X.shape[0]
    n_h = hidden_units
    n_y = Y.shape[0]

    return n_x, n_h, n_y


# ===============[ INITIALISATION DES PARAMÈTRES ]===============
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- taille de la couche d'entrée
    n_h -- taille de la couche cachée
    n_y -- taille de la couche de sortie

    Retourne:
    parameters -- dictionnaire contenant les paramètres du réseau:
                    W1 -- matrice de poids (n_h, n_x)
                    b1 -- vecteur de biais (n_h, 1)
                    W2 -- matrice de poids (n_y, n_h)
                    b2 -- vecteur de biais (n_y, 1)
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2}

    return parameters


# ===============[ PROPAGATION AVANT ]===============
def forward_propagation(X, parameters):
    """
    Arguments:
    X -- dataset d'entrée (nombre_exemples, nombre features)
    parameters -- dictionnaire contenant les paramètres du réseau

    Retourne:
    A2 -- vecteur des activations sigmoïde de chaque exemple
    cache -- dictionnaire contenant Z1, A1, Z2 et A2
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2}

    return A2, cache


# ===============[ CALCUL DU COÛT D'ENTROPIE CROISÉE ]===============
def compute_cost(A2, Y):
    """
    Arguments:
    A2 -- vecteur des activations sigmoïde de chaque exemple (nombre_exemples, 1)
    Y -- étiquettes du dataset (nombre_exemples, nombre sorties)

    Retourne:
    cost -- coût défini par l'équation d'entropie croisée
    """

    m = Y.shape[0]
    correct_classifications = 0

    for i in range(m):
        if (A2[i, 0] >= 0.5 and Y[i, 0] == 1) or (A2[i, 0] < 0.5 and Y[i, 0] == 0):
            correct_classifications += + 1
    efficiency = (correct_classifications / m) * 100

    logprobs = np.multiply(np.log(A2), Y)
    cost = - np.sum(logprobs)

    return cost, efficiency


# ===============[ PROGAGATION ARRIÈRE ]===============
def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- dictionnaire contenant les paramètres du réseau
    cache -- dictionnaire contenant Z1, A1, Z2 et A2
    X -- dataset d'entrée (nombre_exemples, nombre features)
    Y -- étiquettes du dataset (nombre_exemples, nombre sorties)

    Retourne:
    gradients -- dictionnaire contenant les dérivées de chaque paramètre
    """
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2}

    return gradients


# ===============[ MISE À JOUR DES PARAMÈTRES ]===============
def update_parameters(parameters, grads, learning_rate=1.):
    """
    Arguments:
    parameters -- dictionnaire contenant les paramètres du réseau
    gradients -- dictionnaire contenant les dérivées de chaque paramètre

    Returns:
    parameters -- dictionnaire contenant les paramètres du réseau mis à jour
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2}

    return parameters


# ===============[ ASSEMBLAGE DU MODÈLE ]===============
def nn_model(X, Y, hidden_units, num_iterations=1000, learning_rate=1., print_cost=False):
    """
    Arguments:
    X -- dataset d'entrée (nombre_exemples, nombre features)
    Y -- étiquettes du dataset (nombre_exemples, nombre sorties)
    n_h -- taille de la couche cahée
    num_iterations -- nombre d'itérations de la descente de gradient
    print_cost -- affichage du coût toutes les 1000 itérations

    Retourne:
    parameters -- dictionnaire contenant les paramètres appris par le modèle
    cost_history -- liste de coût à différentes étapes d'entraînement
    index_history -- lsite du nombre d'itération à chaque sauvegarde du coût
    """

    cost_history, index_history = [], []

    n_x, n_h, n_y = layer_sizes(X, Y, hidden_units)
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations + 1):

        A2, cache = forward_propagation(X, parameters)
        cost, efficiency = compute_cost(A2, Y)

        gradients = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if print_cost and i % 100 == 0:
            print(f'COÛT ({i} itérations): \t {cost} ({efficiency}% de classifications correctes)')
            cost_history.append(cost)
            index_history.append(i)

    return parameters, cost_history, index_history


# ===============[ APPRENTISSAGE DU MODÈLE ]===============
learnt_parameters, cost_history, index_history = nn_model(X, Y, hidden_units=16, num_iterations=1000, learning_rate=1, print_cost=True)