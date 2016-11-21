from __future__ import division
import numpy as np
import math

class Network(object):
    def __init__(self, input_dim, numHidden=3, learningRate=3):
        self.learningRate, self.numLayers, self.height = learningRate, numHidden + 1, input_dim
        self.W = self.initializeWeights()
        self.outLayer = np.random.rand(self.height + 1)

    def initializeWeights(self):
        weights = np.random.rand(self.numLayers, self.height, self.height + 1)
        # print "\nWEIGHTS\n{}\n\n".format(weights)
        return weights

    def compute(self, x, allActivations=False):
        X = np.array([x])
        for layer in self.W:
            y = np.array([np.dot(x, w[1:]) + w[0] for w in layer])
            y = np.reciprocal(np.add(np.exp(np.negative(y)), 1))
            if allActivations:
                X = np.concatenate((X, np.transpose([[yi] for yi in y])))
            x = y
        out = np.dot(x, self.outLayer[1:]) + self.outLayer[0]
        out = np.reciprocal(np.add(np.exp(np.negative(out)), 1))
        # print out, out.shape
        if allActivations:
            return out, X
        return out

    def backprop(self, x, y):
        out, activations = self.compute(x, allActivations=True)
        delta = np.empty((self.numLayers, self.height))
        dCost = np.subtract(out, y)
        dActivations = np.multiply(activations, np.subtract(1, activations))
        deltaOut = np.multiply(dCost, out*(1-out))
        delta[self.numLayers-1] = np.multiply(np.dot(np.transpose(self.outLayer[1:]), deltaOut), dActivations[self.numLayers])
        for layer in reversed(range(self.numLayers-1)):
            delta[layer] = np.multiply(np.dot(np.transpose(self.W[layer+1,:,1:]), delta[layer+1]), dActivations[layer+1])
        self.outLayer[0] = np.subtract(self.outLayer[0], deltaOut)
        self.outLayer[1:] = np.subtract(self.outLayer[1:], np.multiply(deltaOut, activations[self.numLayers]))
        self.W[:,:,0] = np.subtract(self.W[:,:,0], delta)
        for layer in range(self.numLayers):
            for j in range(self.height):
                self.W[layer,j,1:] = np.subtract(self.W[layer,j,1:], np.multiply(self.learningRate * delta[layer,j], activations[layer]))
        # self.W[:,:,1:] = np.subtract(self.W[:,:,1:], self.learningRate * np.multiply(activations[:-1], delta))
        # self.W[l,j,k] = np.subtract(self.W[l,j,k], self.learningRate * np.multiply(delta[l,j], activations[l,k]))


    def backpropMultiple(self, X, Y):
        indeces = np.arange(len(X))
        np.random.shuffle(indeces)
        for i in indeces:
            self.backprop(X[i], Y[i])

    def trainBatch(self, X, Y, batchsize):
        indeces = np.random.choice(len(X), batchsize, replace=False)
        for i in indeces:
            self.backprop(X[i], Y[i])

    def trainBatches(self, X, Y, batchsize, iterations):
        for i in xrange(iterations):
            self.trainBatch(X, Y, batchsize)

    def test(self, x, y):
        return np.linalg.norm(np.subtract(y, self.compute(x)))

    def testMultiple(self, X, Y):
        error = 0
        for x,y in zip(X,Y):
            error += self.test(x, y)
        return error/len(X)


dim = 3
data = np.random.rand(500, dim)
func = np.random.rand(dim)
print func
out = np.reciprocal(np.add(np.exp(np.negative(np.dot(data, func))), 1))
print out.shape

test = np.random.rand(30, dim)
testOut = np.reciprocal(np.add(np.exp(np.negative(np.dot(test, func))), 1))

beforeTests = []
afterTests = []
print "____COMPUTING:\n",
for rate in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
    net = Network(dim, learningRate=rate)
    print rate
    # print net.W, net.outLayer
    beforeTests.append(net.testMultiple(test, testOut))
    net.trainBatches(data, out, 20, 500)
    # print "new"
    # print net.W, net.outLayer
    afterTests.append(net.testMultiple(test, testOut))
improvement = np.subtract(beforeTests, afterTests)
print improvement
