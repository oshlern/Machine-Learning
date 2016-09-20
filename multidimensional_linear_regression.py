print "-------------Booting up"
import numpy, math, random
from sklearn import linear_model
print "-------------Libraries imported"

x_dim = 1
mx_dim = 2 #dimension of y polynomial
min_x = -10
max_x = 10
length = 5
weights = [random.random()*4-2 for i in range(mx_dim+1)]

def f(x):
    out = 0
    for i in range(len(weights)):
        out += x**i * weights[i]
    return out
    # return x**3 - 2 * x**2 + x - 1

def generateData(f, mx_dim=1, x_dim=1, min_x=5, max_x=25, length=20):
    noise = numpy.random.normal(size=(length, x_dim))
    x = numpy.random.uniform(low=min_x, high=max_x, size=(length, x_dim))
    # print "-------------X:    ", x
    y = [[f(x[i][j])+noise[i][j] for j in range(x_dim)] for i in range(len(x))]
    mx = [[element**dim for element in index for dim in range(1,mx_dim+1)] for index in x]
    return x, mx, y

x, mx, y = generateData(f, mx_dim, x_dim, min_x, max_x, length)
print "-------------Data Generated"
print "-------------X:  ", x
print "-------------MX:  ", mx
print "-------------Y:  ", y

model = linear_model.LinearRegression()
model.fit(mx, y) #not in right format
print "-------------Model Trained"

# error = numpy.sqrt(numpy.average(numpy.square(numpy.subtract(y, model.predict(x)))))
# predicted = model.predict(x)
# print "-------------Model Prediction:", predicted
# print "-------------Error:", error

if x_dim <= 2:
    import matplotlib.pyplot

    # x is an array of arrays, necessary for the model prediction.
    # The [:,0] slicing pulls out the values into a one-dimensional list for plotting.
    x_1d = x[:,0]

    # if this a a 3D plot
    if x_dim == 2:
        from mpl_toolkits.mplot3d import Axes3D

        x2_1d = x[:,1]

        # get the current axes, and tell them to do a 3D projection
        fig = matplotlib.pyplot.figure()
        axes = fig.gca(projection='3d')

        # put the generated points on the graph
        axes.scatter(x_1d, x2_1d, y)

        # predict for input points across the graph to find the best-fit plane
        # and arrange them into the gride that matplotlib appears to require
        X1 = X2 = numpy.arange(min_x, max_x, 0.05)
        X1, X2 = numpy.meshgrid(X1, X2)
        Y = numpy.array(model.predict(zip(X1.flatten(), X2.flatten()))).reshape(X1.shape)

        # put the predicted plane on the graph
        axes.plot_surface(X1, X2, Y, alpha=0.1)

    # if this is a 2D plot
    else:
        # put the generated points on the graph
        matplotlib.pyplot.scatter(x_1d, y)

        # predict for inputs along the graph to find the best-fit line
        X = numpy.linspace(min_x, max_x)
        Y = model.predict(zip(X))
        matplotlib.pyplot.plot(X, Y)

    matplotlib.pyplot.show()
# if x_dim == 1:
#     import matplotlib.pyplot
#     matplotlib.pyplot.scatter(x, y)
#     # predicting the two end points is sufficient to get the whole best fit line
#     matplotlib.pyplot.plot(x, y)
#     matplotlib.pyplot.show()
