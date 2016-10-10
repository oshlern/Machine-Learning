# see basic example here:
#    http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# full documentation of the linear_model module here:
#    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

import numpy as np
from sklearn import linear_model

MIN_X = -10
MAX_X = 10
NUM_INPUTS = 20
NUM_TESTS = 10
noise = np.random.normal(size=NUM_INPUTS) #change for test?
x = np.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS+NUM_TESTS, 1))
x_1d = x[:,0]
x, test, x_1d, test_1d = x[:NUM_INPUTS], x[NUM_INPUTS:], x_1d[:NUM_INPUTS], x_1d[NUM_INPUTS:]


def f(x):
    out = []
    noise = np.random.normal(size=len(x))
    for num, n in zip(x, noise):
        out.append(0.4*x + 0.8 + n)
        #if type(num) == list()
    return out

# y, y_test = f(x_1d), f(test_1d)

# y = 0.3x + 1
y = 0.3 * x_1d + 1 + noise
y_test = 0.3 * test_1d + 1

# y = 0.7x^2 - 0.4x + 1.5
# y, y_test = 0.7 * x_1d * x_1d - 0.4 * x_1d + 1.5 + noise, 0.7 * test_1d * test_1d - 0.4 * test_1d + 1.5 + noise
# y = 0.7 * x_1d * x_1d - 0.4 * x_1d + 1.5 + noise
# y_test = 0.7 * test_1d * test_1d - 0.4 * test_1d + 1.5

### 2 features

# randomly pick 50 pairs of numbers
x2 = np.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 2))

# x is an array of arrays (necessary for the model fit function). The [:,n] slicing pulls out each of
# the values for the nth feature into a one-dimensional list for y

# y = 0.5x_1 - 0.2x_2 - 2
y2 = 0.5 * x2[:,0] * x2[:,0] - x2[:,0] - 0.2 * x2[:,1] * x2[:,1] - 2 + noise



### Select which dataset we are using
# x = x1
# y = y1


################################################################################
# MODEL TRAINING
################################################################################

model = linear_model.LinearRegression()
model.fit(x, y)

# print "test", test
# print model.predict(test)
# error = np.sqrt(np.average(np.square(np.subtract(y, model.predict(x)))))
def rmse(y, predicted):
    err = np.subtract(y, predicted)
    avgSq = np.average(np.square(err))
    return np.sqrt(avgSq)

def rSq(y, predicted):
    yAvg = np.average(y)
    average_y = np.array([yAvg]*len(y))
    errSq = np.square(np.subtract(y, predicted))
    varSq = np.square(np.subtract(y, average_y))
    return 1 - np.average(errSq)/np.average(varSq)

print "________RMSE_______", rmse(y_test, model.predict(test))
print "________R^2_______", rSq(y_test, model.predict(test))

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
        X1 = X2 = np.arange(MIN_X, MAX_X, 0.05)
        X1, X2 = np.meshgrid(X1, X2)
        Y = np.array(model.predict(zip(X1.flatten(), X2.flatten()))).reshape(X1.shape)

        # put the predicted plane on the graph
        axes.plot_surface(X1, X2, Y, alpha=0.1)

    # if this is a 2D plot
    else:
        # put the generated points on the graph
        matplotlib.pyplot.scatter(x_1d, y)

        # predict for inputs along the graph to find the best-fit line
        X = np.linspace(MIN_X, MAX_X)
        Y = model.predict(zip(X))
        matplotlib.pyplot.plot(X, Y)

    matplotlib.pyplot.show()
