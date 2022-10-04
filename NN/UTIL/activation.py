import numpy as np

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x)*(1-sigmoid(x))