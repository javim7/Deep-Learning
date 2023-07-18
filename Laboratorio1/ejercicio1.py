#importando numpy
import numpy as np

'''
Funciones
'''

#Funcion de activacion: Sigmoide
def sigmoid(z):
    # se calcula la funcion sigmoide
    return 1/(1+np.exp(-z))

#Funcion de inicializacion de parametros
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2) # se establece una semilla para que los valores aleatorios no cambien
    W1 = np.random.randn(n_h, n_x) * 0.01 # matriz de pesos de la capa 1
    b1 = np.zeros(shape=(n_h, 1)) # vector bias de la capa 1
    W2 = np.random.randn(n_y, n_h) * 0.01 # matriz de pesos de la capa 2
    b2 = np.zeros(shape=(n_y, 1)) # vector bias de la capa 2

    # se guardan los parametros en un diccionario
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    # se retorna el diccionario
    return parameters

#Funcion de propagacion hacia adelante
def forward_prop(X, parameters):
    # se obtienen los parametros del diccionario
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    # se calcula la propagacion hacia adelante
    Z1 = np.dot(W1, X) + b1 # se calcula el producto punto de W1 y X y se le suma b1
    A1 = np.tanh(Z1) # se calcula la tangente hiperbolica de Z1
    Z2 = np.dot(W2, A1) + b2 # se calcula el producto punto de W2 y A1 y se le suma b2
    A2 = sigmoid(Z2) # se calcula la funcion sigmoide de Z2

    # se guardan los valores de A1 y A2 en un diccionario
    cache = {"A1": A1,
             "A2": A2}

    # se retorna el diccionario
    return A2, cache
    
#Funcion de costo
def calculate_cost(A2, Y):
    m = Y.shape[1] # se obtiene el numero de ejemplos
    cost = (-1/m) * np.sum(Y * np.log(A2) + (1-Y) * (np.log(1-A2))) # se calcula el costo

    # se retorna el costo
    return cost

#funcion de retropropagacion
def backward_prop(X, Y, cache, parameters):
    m = X.shape[1] # se obtiene el numero de ejemplos
    
    # se obtienen los parametros del diccionario
    A1 = cache["A1"]
    A2 = cache["A2"]

    # se obtienen los parametros del diccionario
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # se guardan los valores de dW1, db1, dW2 y db2 en un diccionario
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    # se retorna el diccionario
    return grads

"""
Main
"""
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # matriz de entrada
prueba = forward_prop(X, initialize_parameters(2, 2, 1))
print(prueba)

error = calculate_cost(prueba[0], np.array([[0, 1, 1, 0]]))

print(error)