# Using example code from http://scikit-learn.org/stable/modules/tree.html

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import os.path
import matplotlib.pyplot

# Load the iris dataset
iris = load_iris()

# Train the model
model = tree.DecisionTreeClassifier()
model.fit(iris.data, iris.target)

# create the flow chart visualizaton and write it to a PDF on the Desktop
dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names,
                     filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_pdf(os.path.expanduser("~/Desktop/iris_decision_tree.pdf"))


# Plot two of the features (the first and fourth columns, in this case)
X_INDEX = 0
Y_INDEX = 3

x = iris.data[:,X_INDEX]
y = iris.data[:,Y_INDEX]

# The data are in order by type. Find out where the other types start
start_type_one = list(iris.target).index(1)
start_type_two = list(iris.target).index(2)

# put the input data on the graph, with different colors and shapes for each type
matplotlib.pyplot.scatter(x[:start_type_one], y[:start_type_one], c="red", marker="o")
matplotlib.pyplot.scatter(x[start_type_one:start_type_two], y[start_type_one:start_type_two], c="blue", marker="^")
matplotlib.pyplot.scatter(x[start_type_two:], y[start_type_two:], c="yellow", marker="*")

# Label the axes
matplotlib.pyplot.xlabel(iris.feature_names[X_INDEX])
matplotlib.pyplot.ylabel(iris.feature_names[Y_INDEX])

matplotlib.pyplot.show()
