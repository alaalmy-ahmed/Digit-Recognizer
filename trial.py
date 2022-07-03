import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.models import Sequential


(X_train, y_train), (val, valc) = mnist.load_data()

train = pd.read_csv("train.csv")
y_test = train["label"]
x_test = train.drop(labels = ["label"],axis = 1) 
x_test = x_test / 255.0
x_test = x_test.values.reshape(-1,28,28,1)

x_train = np.concatenate((X_train, val), axis=0)
y_train = np.concatenate((y_train, valc), axis=0)


# X_train = X_train / 255.0
# X_test = X_test / 255.0

# Batch sizes for training and testing
batch_size = 64
test_batch_size = 14

# Training epochs (usually 10 is a good value)
n_epochs = 30

# Learning rate
learning_rate = 1.0

# Decay rate for adjusting the learning rate
gamma = 0.7

# Number of target classes in the MNIST data
num_classes = 10

# Data input shape
input_shape = (28, 28, 1)

# The scaled mean and standard deviation of the MNIST dataset (precalculated)
data_mean = 0.1307
data_std = 0.3081

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], 
                          x_train.shape[1], 
                          x_train.shape[2], 1)

x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2], 1)

# Normalize the data
x_train = (x_train/255.0 - data_mean) / data_std
x_test = (x_test/255.0 - data_mean) / data_std

# Convert labels to one-hot vectors
y_train = tf.one_hot(y_train.astype(np.int32), depth=num_classes)
y_test = tf.one_hot(y_test.astype(np.int32), depth=num_classes)

# Define the architecture of the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
                                      padding='valid', 
                                      activation='relu',
                                      input_shape=input_shape),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1),
                                      padding='valid',
                                      activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])



    # Decay the learning rate at a base rate of gamma roughly every epoch, which
# is len(x_train) steps
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=len(x_train),
    decay_rate=gamma)

# Define the optimizer to user for gradient descent
optimizer = tf.keras.optimizers.Adadelta(scheduler)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display a model summary
# model.summary()

# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_data=(x_test, y_test),
          validation_batch_size=test_batch_size)

model.save_weights("mnist_cnn_tf.ckpt")



test = pd.read_csv("test.csv")


# Drop 'label' column

# Normalize the data
test = (test/255.0 - data_mean) / data_std

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
test = test.values.reshape(-1,28,28,1)

# Split the train and the validation set for the fitting
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("last_test_mad_idea0.csv",index=False)




