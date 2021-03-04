---
layout: single
author_profile: false
---

# Lab 1: Gradient Descent vs Stochastic Gradient Descent, and the Effects of Batch Size
## Gradient Descent
```python
#Generating a random dataset
# Numpy is a fundamental package for scientific computing. It contains an implementation of an array
# that we will use in this exercise.
import numpy as np
# We will be generating our own random dataset. As a consequence we need functionality to generate random numbers.
import random
# We will be plotting the progress of training using matplotlib, a package that can be used to generate 2D and 3D plots.
# We use the "widget" option to enable interactivity later on.
%matplotlib widget
import matplotlib.pyplot as plt
# We will use TensorFlow as the deep learning framework of choice for this class.
import tensorflow as tf

# Define the number of samples/data points you want to generate
n_samples = 100
# We will define a dataset that lies on a line as defined by y = w_gen * x + b_gen
w_gen = 10
b_gen = 2
# To make the problem a bit more interesting we will add some Gaussian noise as 
# defined by the mean and standard deviation below.
mean_gen = 0
std_gen = 1

# This section generates the training dataset as defined by the variables in the section above.
x = np.random.uniform(0, 10, n_samples)
y = np.array([w_gen * (x + np.random.normal(loc=mean_gen, scale=std_gen, size=None)) + b_gen for x in x])

# Plot our randomly generated dataset
plt.close()
plt.plot(x, y, 'go')
plt.xlabel("x", size=24)
plt.ylabel("y", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()

#Defining the model
# Create the placeholders for the data to be used.
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Create our model variables w (weights; this is intended to map to the slope, w_gen) and b (bias; this maps to the intercept, b_gen).
# For simplicity, we initialize the data to zero.
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

# Define our model. We are implementing a simple linear neuron as per the diagram shown above.
Y_predicted = w * X + b

#Defining the loss function
# We define the loss function which is an indicator of how good or bad our model is at any point of time.
loss = tf.reduce_mean(tf.squared_difference(Y_predicted, Y))

#Defining the optimization logic: gradient descent
# Define a gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

#Training loop
# Define the maximum number of times we want to process the entire dataset (the number of epochs).
# In practice we won't run this many because we'll implement an early stopping condition that
# detects when the training process has converged.
max_number_of_epochs = 1000

# We still store information about the optimization process here.
loss_array = []
b_array = []
w_array = []
    
with tf.Session() as sess:

    # Initialize the necessary variables
    sess.run(tf.global_variables_initializer())
    
    # Print out the parameters and loss before we do any training
    w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x, Y: y})
    print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(w_value, b_value, loss_value))
    
    print("")
    print("Starting training")
    print("")

    # Start the training process
    for i in range(max_number_of_epochs):

        # Use the entire dataset to calculate the gradient and update the parameters
        sess.run(optimizer, feed_dict={X: x, Y: y})

        # Capture the data that we will use in our visualization
        w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x, Y: y})
        w_array.append(w_value)
        b_array.append(b_value)
        loss_array.append(loss_value)

        # At the end of every few epochs print out the learned weights
        if (i + 1) % 5 == 0:
            print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, w_value, b_value, loss_value))

        # Implement your convergence check here, and exit the training loop if
        # you detect that we are converged:
        if (i >= 1) and (np.abs(loss_value - loss_array[-2]) / loss_array[-2] < 0.001): # TODO
            break

    print("")
    print("Training finished after {} epochs".format(i+1))
    print("")
    
    print("After training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(w_value, b_value, loss_value))
    
plt.close()
plt.plot(loss_array)
plt.xlabel("Epoch", size=24)
plt.ylabel("Loss", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()

#Investigating the progress of the loss function
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(w_array, b_array, loss_array)

ax.set_xlabel('w', size=16)
ax.set_ylabel('b', size=16)
ax.tick_params(labelsize=12)

plt.show()

loss_surface = []
w_surface = []
b_surface = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for w_value in np.linspace(0, 20, 200):
        for b_value in np.linspace(-18, 22, 200):

            # Collect information about the loss function surface 
            loss_value = sess.run(loss, feed_dict={X: x, Y: y, w: w_value, b: b_value})
            b_surface.append(b_value)
            w_surface.append(w_value)
            loss_surface.append(loss_value)
            
plt.close()

fig = plt.figure()
ax2 = fig.gca(projection='3d')

ax2.scatter(w_surface, b_surface, loss_surface, c = loss_surface, alpha = 0.02)
ax2.plot(w_array, b_array, loss_array, color='black')

ax2.set_xlabel('w')
ax2.set_ylabel('b')

plt.show()
```

## Stochastic Gradient Descent
```python
# Define the maximum number of times we want to process the entire dataset (the number of epochs).
# In practice we won't run this many because we'll implement an early stopping condition that
# detects when the training process has converged.
max_number_of_epochs = 1000

# We still store information about the optimization process here.
loss_array = []
b_array = []
w_array = []
    
with tf.Session() as sess:

    # Initialize the necessary variables
    sess.run(tf.global_variables_initializer())
    
    # Print out the parameters and loss before we do any training
    w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x, Y: y})
    print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(w_value, b_value, loss_value))
    
    print("")
    print("Starting training")
    print("")

    # Start the training process
    for i in range(max_number_of_epochs):

        # Update after every data point
        for (x_pt, y_pt) in zip(x, y):
            sess.run(optimizer, feed_dict={X: x_pt, Y: y_pt})

            # Capture the data that we will use in our visualization
            # Note that we are now updating our loss function after
            # every point in the sample, so the size of loss_array
            # will be greater by a factor of n_samples compared to
            # the last exercise.
            w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x_pt, Y: y_pt})
            w_array.append(w_value)
            b_array.append(b_value)
            loss_array.append(loss_value)

        # At the end of every few epochs print out the learned weights
        if (i + 1) % 5 == 0:
            avg_w = sum(w_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
            avg_b = sum(b_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
            avg_loss = sum(loss_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
            print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, avg_w, avg_b, avg_loss))

        # End the training when the loss function has not changed from the last epoch
        # by more than a small amount. Note that in our convergence check we will compare
        # the loss averaged over this epoch with the loss averaged over the last epoch.
        if i > 1:
            average_loss_this_epoch = sum(loss_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
            average_loss_last_epoch = sum(loss_array[(i-2)*n_samples:(i-1)*n_samples]) / n_samples
            if abs(average_loss_this_epoch - average_loss_last_epoch) / average_loss_last_epoch < 0.001:
                break

    print("")
    print("Training finished after {} epochs".format(i+1))
    print("")
    
    avg_w = sum(w_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
    avg_b = sum(b_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
    avg_loss = sum(loss_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
    
    print("After training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(avg_w, avg_b, avg_loss))
    
plt.close()
plt.plot(loss_array)
plt.xlabel("Number of Updates", size=24)
plt.ylabel("Loss", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(w_array, b_array, loss_array)

ax.set_xlabel('w', size=16)
ax.set_ylabel('b', size=16)
ax.tick_params(labelsize=12)

plt.show()

plt.close()

fig = plt.figure()
ax2 = fig.gca(projection='3d')

ax2.scatter(w_surface, b_surface, loss_surface, c = loss_surface, alpha = 0.02)
ax2.plot(w_array, b_array, loss_array, color='black')

ax2.set_xlabel('w')
ax2.set_ylabel('b')

plt.show()
```
## Optimizing training with batch size
```python

```

# Lab 2: Multi-GPU DL Training Implementation using Horovod

# Lab 3: Algorithmic Concerns for Training at Scale
