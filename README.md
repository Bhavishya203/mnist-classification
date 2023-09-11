# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

<img width="474" alt="dp ex3 model" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/e0fd1ce8-5803-446c-aa8f-a0fa688c4c55">


## DESIGN STEPS

### STEP 1:

Import tensorflow and preprocessing libraries.

### STEP 2:

Download and load the dataset

### STEP 3:

Scale the dataset between it's min and max values

### STEP 4:

Using one hot encode, encode the categorical values

### STEP 5:

Split the data into train and test

### STEP 6:

Build the convolutional neural network model

### STEP 7:

Train the model with the training data

### STEP 8:

Plot the performance plot

### STEP 9:

Evaluate the model with the testing data

### STEP 10:

Fit the model and predict the single input

## PROGRAME
```
Developed By: Bhavishya Reddy Mitta
Register NUmber: 212221230061
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
     
X_train.shape

X_test.shape

single_image= X_train[0]
     
single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
     
model.fit(X_train_scaled ,y_train_onehot, epochs=8,batch_size=128, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

# Prediction for a single input
img = image.load_img('/content/image 5.png')
type(img)

img = image.load_img('/content/image 5.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
     
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="455" alt="dp ex3 fig1" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/d36bac31-dcea-454d-a30b-383f6e5ec316">

<img width="452" alt="dp ex3 fig2" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/dad8d834-f67b-4da0-ab04-d108baa709e6">

### Classification Report

<img width="341" alt="dp ex3 fig3" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/12c1662c-2a28-4956-9f7e-565c7607ff0b">


### Confusion Matrix

<img width="323" alt="dp ex3 fig4" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/78ccee0b-dfd1-488e-803b-1bf06874f67c">


### New Sample Data Prediction

<img width="351" alt="dp ex3 fig5" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/12c76a33-8bcc-43ce-8ca6-84c93b1fbe28">

<img width="188" alt="dp ex3 fig6" src="https://github.com/Bhavishya203/mnist-classification/assets/94679395/2f36ddec-6abc-4f00-8aeb-dcbffef199a6">

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
