from __future__ import division
import numpy as np
import random, re, math
#given x, probInC = probGivenC / probGivenX * probInC,GivenX
def openData(doc):
    text = open(doc, 'r')
    text = text.read()
    return text
def saveData(doc, data):
    text = open(doc, 'w')
    text = text.write(data)
def parse(doc, n):
    text = openData(doc)
    text = re.sub(r'^[a-z0-9\']', '', text.lower())
    text = text.split(' ')
    length = int(math.floor(len(text)/n))
    out = []
    for i in range(n):
        out.append(text[i*length:(i+1)*length])
    return out

classes = parse('big.txt', 6)
# classes = []
# for doc in ['','']
    # classes += [parse(doc)]
# classes = [['a', 'b'], ['a','c','b'], ['d','c','a','e'], ['e','e','e'], ['f','f']]
numClasses = len(classes)
classProbs = [1/numClasses]*numClasses
dataClassProbs = {}
classTotals = [len(i) for i in classes]
total = sum(classTotals)

def dataClassProb(x):
    if x not in dataClassProbs:
        dataClassProbs[x] = []
        for i in range(numClasses):
            count = 0
            for word in classes[i]:
                if word == x:
                    count += 1
            dataClassProbs[x] += [count/classTotals[i]]
    return dataClassProbs[x]

def classify(x):
    xcProb = dataClassProb(x)
    xProb = sum(dataClassProb(x))
    # print xProb
    # print xcProb
    return [(c*xc+random.random()*0.0000000001)/(xProb+random.random()*0.000000000001) for c,xc in zip(classProbs, xcProb)]

def classifyWords(words):
    words = re.sub(r'^[a-z0-9\']', '', words.lower())
    words = words.split(' ')
    out = [0]*numClasses
    for word in words:
        out = [o + w/len(words) for o,w in zip(out, classify(word))]
    return out
print classifyWords('It was close upon four')
# print "e", classify('e')
# print "d", classify('d')
# print "a", classify('a')
print dataClassProbs


words = sum(classes)
def evaluate(test, model):
    truePos, falsePos = 0, 0
    for c in range(len(test)):
        for data in test[c]:
            if model(data) == c:
                truePos += 1
            else:
                falsePos +=1
