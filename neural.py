from __future__ import division
import numpy as np
import math

class Network:
    learningRate = 0.1
    def __init__(input_dim, output_dim, hidden_dim, numHidden=1):
        self.inDim, self.hidDim, self.outDim, self.numHidden = input_dim, hidden_dim, output_dim, numHidden
        self.weights = self.initializeWeights()

    def initializeWeights(self):
        #TODO DONT INITIALIZE TO ZEROS (for gradient descent not being uniform), instead, small random value
        weights = np.array(np.repeat(np.zeroes(self.inDim), self.hidDim, axis=1)) #does work?
        for i in range(self.numHidden-1):
            np.append(weights, np.repeat(np.zeroes(self.hidDim)), self.hidDim, axis=1)
        np.append(weights, np.repeat(np.zeroes(self.hidDim)), self.outDim, axis=1)
        #weights has numHidden many hidden layers + 1 output layer
        return weights

    def efficientCompute(self, x):
        X = np.array(x)
        for layer in self.W:
            y = np.array([np.dot(x, w[0:]) + w[0] for w in layer])
            y = np.reciprocal(np.add(np.exp(np.negate(y))), 1)
            np.append(X, y)
            x = y
        return x, X # output, activationsMatrix

    def efficientBackprop(self, x, y):
        predicted, activations = self.efficientCompute(x) #Save the output of sigmoid? (and use it for dsigmoid)
        #array, numhidden + input + output
        # dCost, delta = self.initializeWeights()
        delta = dw = copy.deepcopy(self.W)
        dCost = np.subtract(predicted, y)
        dActivation = np.dot(activations[self.numHidden+1], np.subtract(1, activations[self.numHidden+1]))
        delta[self.numHidden] = np.multiply(dCost[self.numHidden], dActivation) #TODO: make it stay as array (np function)
        dw[self.numHidden] = np.append(np.multiply(activations[self.numHidden], delta[self.numHidden]), delta[self.numHidden]) # Bias is in front
        # db = delta[self.numHidden]
        for layer in reversed(range(self.numHidden)):
            dActivation = np.dot(activations[layer], np.subtract(1, activations[layer]))
            delta[layer] = np.multiply(np.dot(np.transpose(self.W[layer+1]), delta[layer+1]), dActivation)
            self.W[layer][:,:,0] = np.subtract(self.W[layer][:,:,0], delta[layer])
            self.W[layer][:,:,0:] = np.subtract(self.W[layer][:,:,0:], np.multiply(activations[layer], delta[layer]))
            # self.W[layer] = np.subtract(self.W[layer], delta + dw) #TODO: FIX appending

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
