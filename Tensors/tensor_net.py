import tensorflow as tf
import numpy as np

numLayers = 5
inDim = 3
class Network:
    def __init__(self, numLayers, inDim):
        self.W = tf.Variable(tf.random_uniform([numLayers, inDim, inDim], -0.1, 0.1))
        self.b = tf.Variable(tf.zeros([numLayers, inDim]))

    def feedforward(x):


x_data = np.random.rand(100, inDim).astype(np.float32)
y_data = x_data * x_data * x_data - 0.3 * x_data * x_data + 2 * x_data - 1

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(20001):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]
