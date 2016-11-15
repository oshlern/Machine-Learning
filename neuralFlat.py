from __future__ import division
import numpy as np
import math

class Network(object):
    learningRate = 3 #change
    def __init__(self, input_dim, numHidden=2):
        self.numLayers, self.height = numHidden + 1, input_dim
        self.W = self.initializeWeights()

    def initializeWeights(self):
        weights = np.random.rand(self.numLayers, self.height, self.height + 1)
        print "\nWEIGHTS\n{}\n\n".format(weights)
        return weights

    def efficientCompute(self, x, allActivations=False):
        X = np.array([x])
        for layer in self.W:
            # print "LAYER: ", layer
            y = np.array([np.dot(x, w[1:]) + w[0] for w in layer])
            y = np.reciprocal(np.add(np.exp(np.negative(y)), 1))
            # print np.shape([[yi] for yi in y])
            # print np.shape(X)
            # print y
            # print "X", X
            X = np.concatenate((X, np.transpose([[yi] for yi in y])))
            x = y
        if allActivations:
            return X
        return x

    def efficientBackprop(self, x, y):
        activations = self.efficientCompute(x, allActivations=True)
        delta = np.empty((self.numLayers, self.height))
        dCost = np.subtract(activations[self.numLayers], y)
        dActivations = np.multiply(activations, np.subtract(1, activations))
        delta[self.numLayers-1] = np.multiply(dCost, dActivations[self.numLayers])
        for layer in reversed(range(self.numLayers-1)):
            delta[layer] = np.multiply(np.dot(np.transpose(self.W[layer+1,:,1:]), delta[layer+1]), dActivations[layer+1])
        self.W[:,:,0] = np.subtract(self.W[:,:,0], delta)
        self.W[:,:,1:] = np.subtract(self.W[:,:,1:], self.learningRate * np.multiply(activations[1:], delta))

    def backpropMultiple(self, X, Y):
        indeces = np.arange(len(X))
        np.random.shuffle(indeces)
        for i in indeces:
            self.efficientBackprop(X[i], Y[i])

    def test(self, x, y):
        return np.linalg.norm(np.subtract(y, self.efficientCompute(x)))

    def testMultiple(self, X, Y):
        error = 0
        for x,y in zip(X,Y):
            error += self.test(x, y)
        return error/len(X)


net = Network(3)
data = np.random.rand(5, net.height)
print data[0], data[0]*2
print "____COMPUTING:\n",
original = net.testMultiple(data, data*2)
for i in range(10000):
    if i%1000 == 0:
        print i
    net.backpropMultiple(data, data*2)
final = net.testMultiple(data, data*2)
improvement = original - final
print improvement


    # def cost(predicted, y):
    #     return 0.5(y-predicted)**2
    #     # TODO: average error for batch?
    #
    # def dcost(a, y):
    #     return a - y
    #
    # def backprop(X, error):
    #     # Error = out - y
    #     newError = np.array()
    #     newWeights = np.array()
    #     for layer in np.fliplr(self.W):
    #         newError = np.dot(error, dzW(x, layer))
    #         np.concatenate(newWeights, np.subtract(layer, newError))
    #         error = newError
    #
    #
    # def activation(num):
    #     return 1/(1 + math.e**(-num))
    #     # if num > 1:
    #     #     return 1
    #     # return 0
    #
    # def z(self, x, w):
    #     lin = np.dot(x, w[1:]) + w[0]
    #     return self.activation(lin)
    #
    # def dz(x,w):
    #     eToDot = np.exp(np.dot(x, w[1:]) + w[0])
    #     dlin = np.divide(eToDot, np.square(1 + eToDot))
    #     np.dot(dlin, w[1:])
    #
    # def dzW(x,W):
    #     eToDot = np.exp(np.dot(x, w[1:]) + w[0] for w in W)
    #     dlin = np.divide(eToDot, np.square(1 + eToDot))
    #     return np.array(np.dot(dlin, w[1:]) for w in W)
    #
    # def compute(self, x):
    #     X = np.array(x)
    #     for w in self.W:
    #         y = np.array()
    #         for neuron in w:
    #             numpy.concatenate(y, self.z(x, neuron))
    #         np.concatenate(X, y)
    #         x = y
    #     return x, X
