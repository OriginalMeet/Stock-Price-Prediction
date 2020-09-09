# import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# code to create Training & Testing dataset and saving the data into the
# ‘train_data_RNN.csv’ and ‘test_data_RNN.csv’ in the data directory

"""# importing the RAW dataset
dataset = pd.read_csv('.\\data\\q2_dataset.csv')
print(dataset)

# removing the date & Close/Last column as it will not be used in prediction
dataset = dataset.iloc[:, 2:].values
print(dataset[0:10]) 

#  Creating X and Y
# Where X contains 12 features based on the last 3 days Open, High, Low Prices & Volume
X = []
y = []
# for loop goes from 3 to the last value
for i in range(3, 1259):
    # tempx stores the Open, High, Low & Volume of last 3 days
    tempx = dataset[i-3:i]
    a = []
    # we store each value of the last 3 days and store it in the new array
    for j in range(len(tempx)):
        for k in range(len(tempx[j])):
            a.append(tempx[j][k])
    X.append(a)
    # storing the next day's open Price as the target value
    y.append(dataset[i][1])
# After obtaining the X & y (new dataset) let's convert it into an array
X = np.array(X)
y = np.array(tf.expand_dims(y, axis=-1))

print(X)
print(y)

# randomly splitting the Train & Test data using 70 / 30 Split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=18)

# Converting the X_train & y_train to the dataframe to store the data into a CSV file
train_df = pd.DataFrame(X_train)
train_df['output'] = y_train
print(train_df)

# Converting the X_test & y_test to the dataframe to store the data into a CSV file
test_df = pd.DataFrame(X_test)
test_df['output'] = y_test
print(test_df)

# saving the training data into a CSV file
train_df.to_csv('.\\data\\train_data_RNN.csv')

# saving the test data into CSV file
test_df.to_csv('.\\data\\test_data_RNN.csv')
"""

if __name__ == "__main__":
    # 1. load your training data
    # reading the training dataset
    train_data = pd.read_csv('.\\data\\train_data_RNN.csv')

    # removing index and only selecting the values and storing the values in 2d array
    train_data = train_data.iloc[:, 1:].values

    # preprocessing the dataset using Min_Max Scalling getting all the values between 0 & 1
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = sc.fit_transform(train_data)
    # saving the fitted scaler to use it to change scaling of the test dataset
    pickle.dump(sc, open(".\\models\\min_max_scaling_RNN.pkl", 'wb'))

    # Obtaining X_train & y_train from the train_data
    X_train = train_data_scaled[:, :][:, :-1]
    y_train = train_data_scaled[:, :][:, -1]
    # making y_train the correct dimention
    y_train = np.array(tf.expand_dims(y_train, axis=-1))

    # reshaping the X_train for RNN
    X_train_scaled = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 2. Train your network
    # 		Make sure to print your training loss within training to show progress
    # 		Make sure you print the final training loss
    # model's code to predict the opening price
    model_RNN = tf.keras.models.Sequential([
        # Using GRU's 254 unit, keeping return_sequences=True because I making multilayer RNN with GRUs.
        tf.keras.layers.GRU(units=254, return_sequences=True,
                            input_shape=(X_train.shape[1], 1)),
        # Dropout layer of 0.2 to prevent overfitting
        tf.keras.layers.Dropout(0.2),

        # Another GRU layer of 128 units
        tf.keras.layers.GRU(units=128, return_sequences=True),
        # Dropout layer of 0.2
        tf.keras.layers.Dropout(0.2),

        # GRU layer of 254 untis
        tf.keras.layers.GRU(units=254, return_sequences=True),
        # Dropout layer of 0.2
        tf.keras.layers.Dropout(0.2),

        # GRU layer of 128 untis
        tf.keras.layers.GRU(units=128, return_sequences=True),
        # Dropout layer of 0.2
        tf.keras.layers.Dropout(0.2),

        # GRU layer of 128 units
        tf.keras.layers.GRU(units=128),
        # dropout layer of 0.2
        tf.keras.layers.Dropout(0.2),

        # final output node.
        tf.keras.layers.Dense(units=1)
    ])

    # compiling the modelwith Adam optimiser and mean_squared_error loss function HUber loss
    model_RNN.compile(optimizer='adam', loss=tf.keras.losses.Huber(),metrics=['mse'])
    
    # printing model summary
    model_RNN.summary()

    # training the model on the train dataset
    model_RNN.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

    # 3. Save your model
    model_RNN.save(".\\models\\20833793_RNN_model.h5")
