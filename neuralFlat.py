from __future__ import division
import numpy as np
import math

class Network(object):
    def __init__(self, input_dim, numHidden=2, learningRate=3):
        self.learningRate, self.numLayers, self.height = learningRate, numHidden + 1, input_dim
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
            if allActivations:
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

    def trainBatch(self, X, Y, batchsize):
        indeces = np.random.choice(len(X), batchsize, replace=False)
        for i in indeces:
            self.efficientBackprop(X[i], Y[i])

    def trainBatches(self, X, Y, batchsize, iterations):
        for i in xrange(iterations):
            self.trainBatch(X, Y, batchsize)

    def test(self, x, y):
        return np.linalg.norm(np.subtract(y, self.efficientCompute(x)))

    def testMultiple(self, X, Y):
        error = 0
        for x,y in zip(X,Y):
            error += self.test(x, y)
        return error/len(X)


dim = 3
data = np.random.rand(500, dim)
func = np.random.rand(dim)
print func
out = np.dot(data, func)
print out.shape

test = np.random.rand(30, dim)
testOut = np.dot(test, func)

beforeTests = []
afterTests = []
print "____COMPUTING:\n",
for rate in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
    net = Network(dim, learningRate=rate)
    beforeTests.append(net.testMultiple(test, testOut))
    net.trainBatches(data, out, 20, 100)
    afterTests.append(net.testMultiple(test, testOut))
improvement = np.subtract(beforeTests, afterTests)
print improvement
