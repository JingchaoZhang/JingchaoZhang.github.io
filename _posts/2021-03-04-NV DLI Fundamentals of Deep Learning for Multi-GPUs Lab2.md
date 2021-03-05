---
layout: single
author_profile: false
---

# Lab 2: Multi-GPU DL Training Implementation using Horovod
[Horovod](https://github.com/horovod/horovod) is a distributed deep learning training framework. It is available for TensorFlow, Keras, PyTorch, and Apache MXNet. In this lab you will learn about what Horovod is and how to use it, by distributing across multiple GPUs the training of the classification model we started with in Exercise 3 of Lab 1.

This lab draws heavily on content provided in the [Horovod tutorials](https://github.com/horovod/tutorials).

## Intro to Horovod
Horovod is an open source tool originally [developed by Uber](https://eng.uber.com/horovod/) to support their need for faster deep learning model training across their many engineering teams. It is part of a growing ecosystem of approaches to distributed training, including for example [Distributed TensorFlow](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md). Uber decided to develop a solution that utilized MPI for distributed process communication, and the [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) for its highly optimized implementation of reductions across distributed processes and nodes. The resulting Horovod package delivers on its promise to scale deep learning model training across multiple GPUs and multiple nodes, with only minor code modification and intuitive debugging.

Since its inception in 2017 Horovod has matured significantly, extending its support from just TensorFlow to Keras, PyTorch, and Apache MXNet. Horovod is extensively tested and has been used on some of the largest DL training runs done to date, for example, supporting exascale deep learning on the [Summit system, scaling to over 27,000 V100 GPUs](https://arxiv.org/pdf/1810.01993.pdf).

## Horovod's MPI Roots
Horovod's connection to MPI runs deep, and for programmers familiar with MPI programming, much of what you program to distribute model training with Horovod will feel very familiar. For those unfamiliar with MPI programming, a brief discussion of some of the conventions and considerations required when distributing processes with Horovod, or MPI, is worthwhile.

Horovod, as with MPI, strictly follows the [Single-Program Multiple-Data (SPMD)](https://en.wikipedia.org/wiki/SPMD) paradigm where we implement the instruction flow of multiple processes in the same file/program. Because multiple processes are executing code in parallel, we have to take care about [race conditions](https://en.wikipedia.org/wiki/Race_condition) and also the synchronization of participating processes.

Horovod assigns a unique numerical ID or rank (an MPI concept) to each process executing the program. This rank can be accessed programmatically. As you will see below when writing Horovod code, by identifying a process's rank programmatically in the code we can take steps such as:
* Pin that process to its own exclusive GPU.
* Utilize a single rank for broadcasting values that need to be used uniformly by all ranks.
* Utilize a single rank for collecting and/or reducing values produced by all ranks.
* Utilize a single rank for logging or writing to disk.
As you work through this course, keep these concepts in mind and especially that Horovod will be sending your single program to be executed in parallel by multiple processes. Keeping this in mind will support your intuition and understanding about why we do what we do with Horovod, even though you will only be making edits to a single program.

With Horovod, which can run multiple processes across multiple GPUs, you typically use a single GPU per training process. Part of what makes Horovod simple to use is that it utilizes MPI, and as such, uses much of the MPI nomenclature. The concept of a rank in MPI is of a unique process ID. In this lab we will be using the term "rank" extensively. If you would like to know more about MPI concepts that are utilized heavily in Horovod, please refer to the Horovod documentation.

Schematically, let's look at how MPI can run multiple GPU processes across multiple nodes. Note how each process, or rank, is pinned to a specific GPU:
![Alt text](/images/NV DLI Fundamentals of Deep Learning for Multi-GPUs Lab2 Figure1.PNG?raw=true "Optional Title")  

#Notebook
```python
import horovod.keras as hvd
#Baseline: train the model
!python fashion_mnist.py --epochs 5 --batch-size 512
#Modify the training script
!cp fashion_mnist.py fashion_mnist_original.py
!nvidia-smi
!horovodrun -np 1 python fashion_mnist.py --epochs 1 --batch-size 512
num_gpus = 4
!horovodrun -np $num_gpus python fashion_mnist.py --epochs 1 --batch-size 512
!CUDA_VISIBLE_DEVICES= python fashion_mnist.py --epochs 1 --batch-size 512
```
#Python Script
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
# TODO: Step 1: import Horovod
import horovod.keras as hvd

# TODO: Step 1: initialize Horovod
hvd.init()

# TODO: Step 1: pin to a GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

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

    # TODO: Step 3: Wrap the optimizer in a Horovod distributed optimizer
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
        
    return model

# TODO: Step 2: only set `verbose` to `1` if this is the root worker.
# Otherwise, it should be zero.
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0

# Input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load Fashion MNIST data.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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


callbacks = []
callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))

class PrintThroughput(keras.callbacks.Callback):
    def __init__(self, total_images=0):
        self.total_images = total_images

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.epoch_start_time
        images_per_sec = round(self.total_images / epoch_time, 2)
        print('Images/sec: {}'.format(images_per_sec))

if verbose:
    callbacks.append(PrintThroughput(total_images=len(y_train)))

class StopAtAccuracy(keras.callbacks.Callback):
    def __init__(self, target=0.85, patience=2, verbose=0):
        self.target = target
        self.patience = patience
        self.verbose = verbose
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
        if self.stopped_epoch > 0 and self.verbose == 1:
            print('Early stopping after epoch {}'.format(self.stopped_epoch + 1))

callbacks.append(StopAtAccuracy(target=args.target_accuracy, patience=args.patience, verbose=verbose))

class PrintTotalTime(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time after epoch {}: {}".format(epoch + 1, total_time))

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time: {}".format(total_time))

if verbose:
    callbacks.append(PrintTotalTime())

# TODO: Step 4: broadcast initial variable states from the first worker to 
# all others by adding the broadcast global variables callback.

callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

# TODO: Step 6: average the metrics among workers at the end of every epoch
# by adding the metric average callback.

callbacks.append(hvd.callbacks.MetricAverageCallback())

# Create the model.
model = create_model()

# Train the model.
model.fit_generator(train_iter,
                    # TODO: Step 5: keep the total number of steps the same despite of an increased number of workers
                    steps_per_epoch=len(train_iter) // hvd.size(),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=4,
                    initial_epoch=0,
                    validation_data=test_iter,
                    # TODO: Step 5: set this value to be 3 * num_test_iterations / number_of_workers
                    validation_steps=3 * len(test_iter) // hvd.size())

# Evaluate the model on the full data set.
score = model.evaluate_generator(test_iter, len(test_iter), workers=4)
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
```
