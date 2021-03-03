---
layout: single
author_profile: false
---

At times you will need to clear the GPUs memory, either to reset the GPU state when an experiment goes wrong, or, in between notebooks when you need a fresh start for a new set of exercises.
```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

#MNIST
```python
from tensorflow.keras.datasets import mnist
# the data, split between train and validation sets
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train.shape
x_valid.shape
x_train.dtype
x_train.min()
x_train.max()

import matplotlib.pyplot as plt

image = x_train[0]
plt.imshow(image, cmap='gray')

y_train[0]

#Flatten the image
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)
x_train.shape

#Normalize Data
x_train = x_train / 255
x_valid = x_valid / 255 
x_train.dtype
x_train.min()
x_train.max()

#Categorically Encoding the Labels
import tensorflow.keras as keras
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)
y_train[0:9]

#Instantiating the Model
from tensorflow.keras.models import Sequential
model = Sequential()
from tensorflow.keras.layers import Dense
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)
```

#Image Classification of an American Sign Language Dataset
```python
import pandas as pd
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

#Extracting the Labels and Images
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']
x_train = train_df.values
x_valid = valid_df.values
x_train.shape
y_train.shape
x_valid.shape
y_valid.shape

import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    
    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    
#Normalize the Image Data
x_train.min()
x_train.max()
# TODO: Normalize x_train and x_valid.
x_train = x_train / 255
x_valid = x_valid / 255 

#Categorize the Labels
import tensorflow.keras as keras
num_classes = 24
# TODO: Categorically encode y_train and y_valid.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

#Build the Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# TODO: build a model following the guidelines above.
model = Sequential([
    Dense(units=512, activation='relu', input_shape=(784,)),
    Dense(units = 512, activation='relu'),
    Dense(units = 24, activation='softmax')])
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# TODO: Train the model for 20 epochs.
history = model.fit(
    x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid)
)
#This is an example of the model learning to categorize the training data, but performing poorly against new data that it has not been trained on. Essentially, it is memorizing the dataset, but not gaining a robust and general understanding of the problem. This is a common issue called overfitting. We will discuss overfitting in the next two lectures, as well as some ways to address it.
```

#CNN
```python
#Loading and Preparing the Data
import tensorflow.keras as keras
import pandas as pd

# Load in our data from CSV files
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

# Separate out our target values
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# Separate out our image vectors
x_train = train_df.values
x_valid = valid_df.values

# Turn our scalar targets into binary categories
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Normalize our image data
x_train = x_train / 255
x_valid = x_valid / 255

#Reshaping Images
x_train.shape, x_valid.shape
x_train = x_train.reshape(-1,28,28,1)
x_valid = x_valid.reshape(-1,28,28,1)
x_train.shape
x_valid.shape
x_train.shape, x_valid.shape

#Create a Convolutional Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid))
```

#Data Augmentation and Deployment
```python
```
