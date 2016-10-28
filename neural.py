from __future__ import division
import numpy as np
import math

class network:
    def __init__(input_dim, output_dim, hidden_dim, numHidden=1):
        self.hidDim = hidden_dim
        self.inDim = input_dim
        self.outDim = output_dim
        # W = [np.zeroes(input_dim)]*hidden_dim

    def initializeWeights():
        self.W = np.array(np.repeat(np.zeroes(self.inDim), self.hidDim, axis=1)) #does work?
        for i in range(numHidden-1):
            np.append(self.W, np.repeat(np.zeroes(self.hidDim)), self.hidDim, axis=1)
        np.append(self.W, np.repeat(np.zeroes(self.hidDim)), self.outDim, axis=1)

    def activation(num):
        return 1/(1 + math.e**(-num))
        # if num > 0:
        #     return 1
        # return 0

    def z(x, w):
        lin = np.dot(x, w[0:]) + w[0]
        return self.activation(lin)

    def compute(x):
        X = np.array(x)
        for w in self.W:
            y = np.array()
            for neuron in w:
                numpy.append(y, self.z(x, neuron))
            np.append(X, y)
            x = y
        return x, X

    def efficentCompute(x):
        X = np.array(x)
        for layer in self.W:
            y = np.array([np.dot(x, w[0:]) + w[0] for w in layer])
            y = np.reciprocal(np.add(np.exp(np.negate(y))), 1)
            np.append(X, y)
            x = y
        return x, X

    def cost(predicted, y):
        return (y-predicted)**2

    def backprop(X, error):
