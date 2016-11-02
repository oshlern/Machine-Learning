from __future__ import division
import numpy as np
import math

class network:
    def __init__(input_dim, output_dim, hidden_dim, numHidden=1):
        self.inDim, self.hidDim, self.outDim, self.numHidden = input_dim, hidden_dim, output_dim, numHidden
        self.weights = self.initializeWeights()

    def initializeWeights(self):
        weights = np.array(np.repeat(np.zeroes(self.inDim), self.hidDim, axis=1)) #does work?
        for i in range(self.numHidden-1):
            np.append(weights, np.repeat(np.zeroes(self.hidDim)), self.hidDim, axis=1)
        np.append(weights, np.repeat(np.zeroes(self.hidDim)), self.outDim, axis=1)
        return weights

    def efficientCompute(self, x):
        X = np.array()
        for layer in self.W:
            y = np.array([np.dot(x, w[0:]) + w[0] for w in layer])
            y = np.reciprocal(np.add(np.exp(np.negate(y))), 1)
            np.append(X, y)
            x = y
        return x, X

    def efficientBackprop(self, x):
        error, activations = self.efficientCompute(x) #Save the output of sigmoid? (and use it for dsigmoid)
        delta = self.initializeWeights()
        eDot = np.exp(np.negative(np.dot(x, w[0:]) + w[0] for w in self.W[-layer])) #does this syntax work? (needs parentheses?)
        dActivation = np.divide(eToDot, np.square(np.add(eToDot, 1)))

        delta[self.numHidden] = np.dot(error, dActivation) #TODO: make it stay as array (np function)
        dw = np.append(delta, np.dot(activations[layer-1], delta))
        for layer in range(self.numHidden+1, 0, -1):
            expDot = np.exp(np.dot(x, w[0:]) + w[0] for w in self.W[-layer]) #does this syntax work? (needs parentheses?)
            dActivation = np.divide(eToDot, np.square(np.add(eToDot, 1)))
            delta = np.dot(error, dActivation) #TODO: make it stay as array (np function)
            dw = np.append(delta, np.dot(activations[layer-1], delta))
            self.W[layer] = np.subtract(self.W[layer], delta + dw) #TODO: FIX appending
            error = newError

    def cost(predicted, y):
        return 0.5(y-predicted)**2
        # TODO: average error for batch?

    def dcost(a, y):
        return a - y

    def backprop(X, error):
        # Error = out - y
        newError = np.array()
        newWeights = np.array()
        for layer in np.fliplr(self.W):
            newError = np.dot(error, dzW(x, layer))
            np.append(newWeights, np.subtract(layer, newError))
            error = newError


    def activation(num):
        return 1/(1 + math.e**(-num))
        # if num > 0:
        #     return 1
        # return 0

    def z(self, x, w):
        lin = np.dot(x, w[0:]) + w[0]
        return self.activation(lin)

    def dz(x,w):
        eToDot = np.exp(np.dot(x, w[0:]) + w[0])
        dlin = np.divide(eToDot, np.square(1 + eToDot))
        np.dot(dlin, w[0:])

    def dzW(x,W):
        eToDot = np.exp(np.dot(x, w[0:]) + w[0] for w in W)
        dlin = np.divide(eToDot, np.square(1 + eToDot))
        return np.array(np.dot(dlin, w[0:]) for w in W)

    def compute(self, x):
        X = np.array(x)
        for w in self.W:
            y = np.array()
            for neuron in w:
                numpy.append(y, self.z(x, neuron))
            np.append(X, y)
            x = y
        return x, X
