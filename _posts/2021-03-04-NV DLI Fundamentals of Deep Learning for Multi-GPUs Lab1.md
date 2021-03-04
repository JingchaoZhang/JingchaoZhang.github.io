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
# Define the maximum number of times we want to process the entire dataset (the number of epochs).
# In practice we won't run this many because we'll implement an early stopping condition that
# detects when the training process has converged.
import math
max_number_of_epochs = 1000

# We still store information about the optimization process here.
loss_array = []
b_array = []
w_array = []

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
with tf.Session() as sess:

    # Initialize the necessary variables
    sess.run(tf.global_variables_initializer())

    # Print out the parameters and loss before we do any training
    w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x, Y: y})
    print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(w_value, b_value, loss_value))

    print("")
    print("Starting training")
    print("")

    # Pass in batches of the dataset
    # After the first run, try batch sizes of 16, 64, and 128
    batch_size = 128
    num_batches_in_epoch = math.floor((n_samples + batch_size - 1) / batch_size) #FIXME

    # Start the training process
    for i in range(max_number_of_epochs):

        for (x_batch, y_batch) in zip(list(chunks(x,batch_size)), list(chunks(y,batch_size))): #FIXME in FIXME:
            #sess.run(optimizer, feed_dict={X: FIXME, Y: FIXME})
            sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})

            # Capture the data that we will use in our visualization
            # These should be calculated only with the current batch
            #w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: FIXME, Y: FIXME})
            w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x_batch, Y: y_batch})
            w_array.append(w_value)
            b_array.append(b_value)
            loss_array.append(loss_value)

        # At the end of every few epochs print out the learned weights
        if (i + 1) % 5 == 0:
            avg_w = sum(w_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
            avg_b = sum(b_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
            avg_loss = sum(loss_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
            print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, avg_w, avg_b, avg_loss))

        # End the training when the loss function has not changed from the last epoch
        # by more than a small amount. Note that in our convergence check we will compare
        # the loss averaged over this epoch with the loss averaged over the last epoch.
        if i > 1:
            average_loss_this_epoch = sum(loss_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
            average_loss_last_epoch = sum(loss_array[(i-2)*num_batches_in_epoch:(i-1)*num_batches_in_epoch]) / num_batches_in_epoch
            if abs(average_loss_this_epoch - average_loss_last_epoch) / average_loss_last_epoch < 0.001:
                break

    print("")
    print("Training finished after {} epochs".format(i+1))
    print("")

    avg_w = sum(w_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
    avg_b = sum(b_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
    avg_loss = sum(loss_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch

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

## The Fashion-MNIST Dataset
### The Python Script
```python
from __future__ import print_function

import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.datasets import fashion_mnist
from keras_contrib.applications.wide_resnet import WideResidualNetwork
import numpy as np
import tensorflow as tf
import os
from time import time

# Parse input arguments

parser = argparse.ArgumentParser(description='Keras Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--wd', type=float, default=0.000005,
                    help='weight decay')
# TODO Step 2: Add target and patience arguments to the argument parser
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')

args = parser.parse_args()

# Define a function for a simple learning rate decay over time

def lr_schedule(epoch):
    
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr

# Define the function that creates the model

def create_model():

    # Set up standard WideResNet-16-10 model.
    model = WideResidualNetwork(depth=16, width=10, weights=None, input_shape=input_shape,
                                classes=num_classes, dropout_rate=0.01)

    # WideResNet model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = keras.models.Model.from_config(model_config)

    opt = keras.optimizers.SGD(lr=args.base_lr)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

verbose = 1

# Input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load Fashion MNIST data.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Train only on 1/6 of the dataset
x_train = x_train[:10000,:,:]
y_train = y_train[:10000]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Training data iterator.
train_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                     horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
train_gen.fit(x_train)
train_iter = train_gen.flow(x_train, y_train, batch_size=args.batch_size)

# Validation data iterator.
test_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_gen.mean = train_gen.mean
test_gen.std = train_gen.std
test_iter = test_gen.flow(x_test, y_test, batch_size=args.val_batch_size)

# TODO Step 1: Define the PrintThroughput callback
class PrintThroughput(keras.callbacks.Callback):
    def __init__(self, total_images=0):
        self.total_images = total_images
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()
    
    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.epoch_start_time
        images_per_sec = round(self.total_images / epoch_time, 2)
        print('Images/sec: {}'.format(images_per_sec))

# TODO Step 2: Define the StopAtAccuracy callback
class StopAtAccuracy(keras.callbacks.Callback):
    def __init__(self, target=0.85, patience=2):
        self.target = target
        self.patience = patience
        self.stopped_epoch = 0
        self.met_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_acc') > self.target:
            self.met_target += 1
        else:
            self.met_target = 0
            
        if self.met_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Early stopping after epoch {}'.format(self.stopped_epoch + 1))

# TODO Step 3: Define the PrintTotalTime callback
class PrintTotalTime(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time after epoch {}: {}".format(epoch + 1, total_time))

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time: {}".format(total_time))

callbacks = [PrintThroughput(total_images=len(y_train)),
             StopAtAccuracy(target=args.target_accuracy, patience=args.patience),
             PrintTotalTime()]
callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))

# Create the model.

model = create_model()

# Train the model.
model.fit_generator(train_iter,
                    steps_per_epoch=len(train_iter),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=4,
                    initial_epoch=0,
                    validation_data=test_iter,
                    validation_steps=len(test_iter))

# Evaluate the model on the full data set.
score = model.evaluate_generator(test_iter, len(test_iter), workers=4)
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
```
### The Notebook
```python
!python fashion_mnist.py
!cp fashion_mnist.py fashion_mnist_original.py
!python fashion_mnist.py --epochs 5
!python fashion_mnist.py --epochs 5 --batch-size 700
%matplotlib widget
import matplotlib.pyplot as plt

data = [('8', 328),
        ("16", 551),
        ("32", 808),
        ("64", 1002),
        ("128", 1165),
        ("256", 1273),
        ("512", 1329),
        ("700", 1332)] # See what happens when you go much above 700

x,y = zip(*data)
plt.bar(x,y)
plt.ylabel("Throughput (images / sec)")
plt.xlabel("Batch Size")
plt.show()
!python fashion_mnist.py --target-accuracy .82 --patience 2
!python fashion_mnist.py --batch-size 32 --target-accuracy 0.82 --patience 2
```

# Lab 3: Algorithmic Concerns for Training at Scale
