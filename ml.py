# Notes:
#   - https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/


################################################################################
# IMPORTS
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, \
    LSTM, CuDNNLSTM, GRU, CuDNNGRU
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import shutil


################################################################################
# HYPERPARAMETERS
BATCH_SIZE = 16
NUM_EPOCHS = 10


################################################################################
def delete_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)


################################################################################
# LOAD DATASET
delete_dir(os.path.join(os.getcwd(), ".idea"))
data_file = os.path.join(os.getcwd(), "test_Data.csv")
data = pd.read_csv(data_file, header=None)

num_rows = len(data.index)
#print(num_rows)
num_cols = len(data.columns)

# normalize timestamp values
#data[:][0] = data[:][0] / np.max(data[:][0])
data = data.drop(data.columns[[0]], axis=1)  # remove timestamp column from data

# normalize A, B, C, D values between [0,1]
for i in range(1, num_cols):  # exclude timestamp column
    data[:][i] = data[:][i] / 5

print(data[:5])

################################################################################
# SPLIT DATASET
train_data = np.array(data[:300])
val_data = np.array(data[300:390])
test_data = np.array(data[300:390])

# CREATE INPUTS AND OUTPUTS FOR DATASETS
train_data, train_label = train_data[:-1], train_data[1:]
val_data, val_label = val_data[:-1], val_data[1:]
test_data, test_label = test_data[:-1], test_data[1:]

print(train_data.shape)
print(train_label.shape)

# reshape datasets to feed to GRU
train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
val_data = val_data.reshape((val_data.shape[0], 1, val_data.shape[1]))
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))


################################################################################
# BUILD MODEL using tf.keras
# build model layer by layer using .add() in a sequential manner
def build_model():
    gpu_check = tf.test.is_gpu_available()

    m = Sequential()

    #potentially put in dense or embedding layer

    if gpu_check:
        m.add(CuDNNGRU(
            units=512,
            input_shape=(train_data.shape[1], train_data.shape[2]),  # (batch_size, num_time_steps, num_features)
            return_sequences=True
        ))

    else:  # no GPU
        m.add(GRU(
            units=512,
            input_shape=(train_data.shape[1], train_data.shape[2]),  # (batch_size, num_time_steps, num_features)
            return_sequences=True
        ))

    m.add(Flatten())

    m.add(Dense(
        units=4
    ))

    # configure how model will be trained
    # define loss function and optimizer choices
    m.compile(
        loss=MSE,
        optimizer=Adam(),
        metrics=["accuracy"]
    )

    m.summary()  # prints out architecture of network
    return m


################################################################################
# TRAIN MODEL
model = build_model()  # instantiate model

# create callbacks to save checkpoints and TensorBoard
# create a folder for each model run
save_folder = os.path.join(os.getcwd(), str(datetime.now().strftime("%m-%d-%Y_%H-%M-%S")))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

history_file = save_folder + "\checkpoint.h5"  # file to save training checkpoints after each epoch
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=save_folder)

# train the model using fit()
history = model.fit(
    x=train_data,
    y=train_label,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(val_data, val_label),
    callbacks=[save_callback, tb_callback],
    verbose=1
)

# get training metrics
history_dict = history.history
train_accuracy = history_dict["acc"]
train_loss = history_dict["loss"]
valid_accuracy = history_dict["val_acc"]
valid_loss = history_dict["val_loss"]

# Plot Training loss and accuracy
epoch_range = range(1, NUM_EPOCHS+1)

plt.plot(epoch_range, train_loss)
plt.title("Training Loss")
plt.show()

plt.plot(epoch_range, train_accuracy)
plt.title("Training Accuracy")
plt.show()

################################################################################
# PREDICTIONS
preds = model.predict(test_data)
max = []

'''
# Convert values back to range [1,5]
for row in range(len(preds)):
    for col in range(len(preds[row])):
        preds[row][col] = preds[row][col] * 5

        if(preds[row][col] < 1):
            preds[row][col] = 1

        if(preds[row][col] > 5):
            preds[row][col] = 5

        preds[row][col] = int(preds[row][col])
'''

# find max value of (A,B,C,D)
for i in range(len(preds)):
    max_idx = np.argmax(preds[i])
    max_value = np.max(preds[i])
    max.append((max_idx, max_value))


# PRINT OUT TO CONSOLE
print("\n###################################################")
print("\nSome predicted values (normalized): ", preds[:2])
print("\nSome (index, max value) pairs: ", max[:2])
print("\nSome actual test values: ", test_data[:2])


