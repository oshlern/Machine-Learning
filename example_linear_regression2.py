# see basic example here:
#    http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# full documentation of the linear_model module here:
#    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

import numpy.random
from sklearn import linear_model

MIN_X = -10
MAX_X = 10
NUM_INPUTS = 50

################################################################################
#  GENERATED DATA
################################################################################

# Generate some normally distributed noise
noise = numpy.random.normal(size=NUM_INPUTS)

### 1 feature

# randomly pick 50 numbers
x1 = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 1))

# x is an array of arrays (necessary for the model fit function).
# The [:,0] slicing pulls out the values into a one-dimensional list for y
x1_1d = x1[:,0]

# y = 0.3x + 1
y1_1 = 0.3 * x1_1d + 1 + noise

# y = 0.7x^2 - 0.4x + 1.5
y1_2 = 0.7 * x1_1d * x1_1d - 0.4 * x1_1d + 1.5 + noise


### 2 features

# randomly pick 50 pairs of numbers
x2 = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 2))

# x is an array of arrays (necessary for the model fit function). The [:,n] slicing pulls out each of
# the values for the nth feature into a one-dimensional list for y

# y = 0.5x_1 - 0.2x_2 - 2
y2 = 0.5 * x2[:,0] - 0.2 * x2[:,1] - 2 + noise



### Select which dataset we are using
x = x1
y = y1_1


################################################################################
# MODEL TRAINING
################################################################################

model = linear_model.LinearRegression()
model.fit(x, y)

print 'Intercept: {0}  Coefficients: {1}'.format(model.intercept_, model.coef_)


################################################################################
# PLOT
################################################################################

# Only plot if x is one- or two-dimensional, meaning that we have a 2D or 3D plot
if x.shape[1] <= 2:
    import matplotlib.pyplot

    # x is an array of arrays, necessary for the model prediction.
    # The [:,0] slicing pulls out the values into a one-dimensional list for plotting.
    x_1d = x[:,0]

    # if this a a 3D plot
    if x.shape[1] == 2:
        from mpl_toolkits.mplot3d import Axes3D

        x2_1d = x[:,1]

        # get the current axes, and tell them to do a 3D projection
        fig = matplotlib.pyplot.figure()
        axes = fig.gca(projection='3d')

        # put the generated points on the graph
        axes.scatter(x_1d, x2_1d, y)

        # predict for input points across the graph to find the best-fit plane
        # and arrange them into the gride that matplotlib appears to require
        X1 = X2 = numpy.arange(MIN_X, MAX_X, 0.05)
        X1, X2 = numpy.meshgrid(X1, X2)
        Y = numpy.array(model.predict(zip(X1.flatten(), X2.flatten()))).reshape(X1.shape)

        # put the predicted plane on the graph
        axes.plot_surface(X1, X2, Y, alpha=0.1)

    # if this is a 2D plot
    else:
        # put the generated points on the graph
        matplotlib.pyplot.scatter(x_1d, y)

        # predict for inputs along the graph to find the best-fit line
        X = numpy.linspace(MIN_X, MAX_X)
        Y = model.predict(zip(X))
        matplotlib.pyplot.plot(X, Y)

    matplotlib.pyplot.show()
