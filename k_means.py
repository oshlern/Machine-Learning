from __future__ import division
import math, random, copy #, numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

# Replace weird array operations with numpy
# floor division //, normal division /?
numClusters = 3
xPerCluster = 60 #Make variable size (with arrays or just refinding length each time)
variation = 10
max_x = 100
dim = 3
threshold = 0.01
max_iterations = 50
X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2

FIGURES_PER_ROW = 3
MAX_CLUSTERS = 9
def generateClusters(numClusters=3):
    return [[max_x * (random.random()*0.5 + 0.25) for d in xrange(dim)] for i in range(numClusters)]

def categorize(data, clusters):
    categorized = []
    for x in data:
        closest, closestCluster = [(clusters[0][i]-x[i])**2 for i in range(dim)], 0
        for i in range(len(clusters)):
            dist = [(clusters[i][j]-x[j])**2 for j in range(dim)]
            if dist < closest:
                closest, closestCluster = dist, i
        categorized.append(closestCluster)
    return categorized

def generateClustersX(x, batchSize, numClusters=3):
    tempX = copy.deepcopy(x)
    tempClusters = [[0]*dim]*numClusters
    print x
    print len(tempX)
    for n in range(numClusters):
        for i in range(batchSize):
            dataPoint = tempX.pop(random.randrange(0,len(tempX)))
            print dataPoint
            tempClusters[n] = [tempClusters[n][j] + dataPoint[j] for j in range(dim)]
        for i in range(numClusters):
            tempClusters[i] = [clusterX/batchSize for clusterX in tempClusters[i]]
    categorized = categorize(x, tempClusters)
    for i in range(numClusters):
        if i not in categorized:
            print i
    print categorized
            # return generateClustersX(x,batchSize,numClusters)
    return tempClusters

def generateData(numClusters=3):
    cs = generateClusters(numClusters=numClusters)
    x = [[[cx + variation * random.random() for cx in c] for i in range(xPerCluster)] for c in cs]
    return x, cs

trueX, trueClusters = generateData(numClusters=numClusters)
x = [] #sum(trueX)
for cluster in trueX:
    x += cluster

#
# def distance(x, cluster):
#     return sum([(ci-xi)**2 for ci,xi in zip(x, cluster)])


    #Change to be arrays of data where each array is a cluster?

# def evaluate(categorized, trueX, clusters):
#     right, wrong = 0, 0
#     for i in range(len(categorized)):
#         if categorized[i] == i//xPerCluster:
#             right += 1
#         else:
#             wrong += 1
#     return right/(right+wrong)

def iterate(x, categorized, numClusters=3, randomness=0):
    tempClusters = [[0]*dim]*numClusters
    count = [0]*numClusters
    for i in range(len(categorized)):
        tempClusters[categorized[i]] = [tempClusters[categorized[i]][j] + x[i][j] + randomness for j in range(dim)]
        count[categorized[i]] += 1
    for i in range(numClusters):
        if count[i] != 0:
            tempClusters[i] = [clusterX/count[i] for clusterX in tempClusters[i]]
    return tempClusters

def clusterChange(oldClusters, newClusters):
    change = 0
    for oldCluster, newCluster in zip(oldClusters, newClusters):
        for oldAxis, newAxis in zip(oldCluster, newCluster):
            change += (newAxis - oldAxis)**2
    return change

def solve(x, numClusters):
    tempClusters = generateClustersX(x, len(x)//10, numClusters=numClusters)
    categorized = categorize(x, tempClusters)
    print "initial clusters", tempClusters, categorized
    newClusters = iterate(x, categorized)
    change = clusterChange(tempClusters, newClusters)
    changeInChange = 1
    while change > threshold:
        tempClusters = newClusters
        categorized = categorize(x, tempClusters)
        newClusters = iterate(x, categorized)
        print tempClusters == newClusters
        newChange = clusterChange(tempClusters, newClusters)
        if (newChange - change) == 0:
            print "no change in ", change
            return tempClusters, categorized
        print "changeInChange", (newChange - change)
        change = newChange
        # print change
    return tempClusters, categorized

# print solve(x, 3)
# print "true:"
# print trueClusters
# print "xperc", xPerCluster



# def evaluateDist(categorized, x, clusters):



def add_plot(figure, subplot_num, subplot_name, data, labels):
    '''Create a new subplot in the figure.'''

    # create a new subplot
    axis = figure.add_subplot(FIGURES_PER_ROW, MAX_CLUSTERS / FIGURES_PER_ROW, subplot_num, projection='3d',
                              elev=48, azim=134)

    # Plot three of the four features on the graph, and set the color according to the labels
    axis.scatter([dataPoint[0] for dataPoint in data], [dataPoint[1] for dataPoint in data], [dataPoint[2] for dataPoint in data], c=labels)

    # get rid of the tick numbers. Otherwise, they all overlap and it looks horrible
    axis.w_xaxis.set_ticklabels([])
    axis.w_yaxis.set_ticklabels([])
    axis.w_zaxis.set_ticklabels([])

    # label the features
    # I found this made the plot a bit too messy for my taste, but you can put the labels back in.
    #axis.set_xlabel(feature_names[X_INDEX])
    #axis.set_ylabel(feature_names[Y_INDEX])
    #axis.set_zlabel(feature_names[Z_INDEX])

    # label the subplot
    axis.title.set_text(subplot_name)

# start a new figure to hold all of the subplots
figure = matplotlib.pyplot.figure(figsize=(4*FIGURES_PER_ROW, MAX_CLUSTERS))

# Plot the ground truth
labeled = []
for i in range(numClusters):
    for j in xrange(xPerCluster):
        labeled.append(i)
add_plot(figure, 1, "Ground Truth", x, labeled)



def solveShow(x, numClusters):
    clusterPath = []
    tempClusters = generateClustersX(x, 5, numClusters=numClusters)
    categorized = categorize(x, tempClusters)
    initCategorized = categorized
    add_plot(figure, 2, "Initialized", x, categorized)
    newClusters = iterate(x, categorized)
    change = clusterChange(tempClusters, newClusters)
    counter = 0
    while counter < max_iterations and newClusters != tempClusters:
        clusterPath += tempClusters
        tempClusters = newClusters
        categorized = categorize(x, tempClusters)
        # add_plot(figure, counter+3, "Initialized", x, categorized)
        newClusters = iterate(x, categorized, randomness=1)
        # change = clusterChange(tempClusters, newClusters)
        # print "change", change
        counter += 1
    if counter > max_iterations:
        print counter, "ALERT"
        return solveShow(x,numClusters)
    add_plot(figure, 2, "Initialized", x, initCategorized)
    add_plot(figure, 3, "10 Hours Later", x, categorized)
    # print categorized
    # print "____X___", x
    return tempClusters, counter, clusterPath, categorized
clusts, counter, clusterPath, categorized = solveShow(x, numClusters)
# print "counter", counter
# print "clusterPath_____  ", clusterPath
labeledClusterPath = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]*len(clusterPath)
# print labeledClusterPath
# print len(x)
# print len(clusterPath)

add_plot(figure, 9, "ClusterPath", clusterPath, labeledClusterPath)
matplotlib.pyplot.show()

# def generateData(x_dim=2, variation=0, max_x=50, length=20, numClusters=3):
#     cs = max_x * numpy.random.rand(size=(numClusters, x_dim))
#     noise = numpy.random.normal(size=(length, x_dim))
#     x = [numpy.random.uniform(low=-variation, high=variation, size=(length, x_dim)) + noise for c in cs]
#     return x
