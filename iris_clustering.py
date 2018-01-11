from __future__ import division
# -*- coding: utf-8  -*-

####################################################################################################
# Documentation of varioius clustering methods: http://scikit-learn.org/stable/modules/clustering.html
#
# Code based on example http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
#
# Original had following comments:
# Code source: Gael Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
####################################################################################################

import numpy
# import matplotlib.pyplot
# from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets

# We can only plot 3 of the 4 features, since we only see in 3D.
# These are the ones the example code picked
X_INDEX = 3
Y_INDEX = 0
Z_INDEX = 2

FIGURES_PER_ROW = 3
MAX_CLUSTERS = 6

def add_plot(figure, subplot_num, subplot_name, data, feature_names, labels):
    '''Create a new subplot in the figure.'''

    # create a new subplot
    axis = figure.add_subplot(FIGURES_PER_ROW, MAX_CLUSTERS / FIGURES_PER_ROW, subplot_num, projection='3d',
                              elev=48, azim=134)

    # Plot three of the four features on the graph, and set the color according to the labels
    axis.scatter(data[:, X_INDEX], data[:, Y_INDEX], data[:, Z_INDEX], c=labels)

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



iris = datasets.load_iris()
# Validate: compare each pair of points (same cluster or different)
clusters = {}
for i in range(len(iris.target)):
    if iris.target[i] in clusters:
        clusters[iris.target[i]]
# print iris.data
# print "_____TARGET_____"
# print iris.target

# start a new figure to hold all of the subplots
# figure = matplotlib.pyplot.figure(figsize=(4*FIGURES_PER_ROW, MAX_CLUSTERS))

# Plot the ground truth
# add_plot(figure, 1, "Ground Truth", iris.data, iris.feature_names, iris.target)

# Train models with different numbers of clusters and put each one in a subplot
for num_clusters in xrange(2, MAX_CLUSTERS + 1):
    # train the model
    model = KMeans(n_clusters=num_clusters)
    model.fit(iris.data)
    # get the predictions
    labels = model.labels_
    # add_plot(figure, num_clusters, '{} Clusters'.format(num_clusters),
            #  iris.data, iris.feature_names, labels.astype(numpy.float))
    length = len(labels)
    tp, fp, tn, fn = 0, 0, 0, 0 #[], [], [], []
    for i in range(length):
        for j in range(i + 1, length):
            isSame = (iris.target[i] == iris.target[j])
            newIsSame = (labels[i] == labels[j])
            if isSame:
                if newIsSame:
                    tp += 1 #.append([i,j])
                else:
                    fn += 1 #.append([i,j])
            elif newIsSame:
                fp += 1 #.append([i,j])
            else:
                tn += 1 #.append([i,j])
    print num_clusters
    # print "___LABELS", labels#.astype(numpy.float)
    print "True Positives: ", tp, "     False Positives: ", fp
    print "True Negatives: ", tn, "     False Negatives: ", fn
    print "Percent True:  ", (tp + tn)/length
# matplotlib.pyplot.show()
