# import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle


if __name__ == "__main__":
    # 1. Load your saved model
    new_model = tf.keras.models.load_model(".\\models\\20833793_RNN_model.h5")

    sc = pickle.load(open(".\\models\\min_max_scaling_RNN.pkl", 'rb'))

    # 2. Load your testing data
    test_data = pd.read_csv('.\\data\\test_data_RNN.csv')

    # print(test_data)
    test_data = test_data.iloc[:, 1:].values
    # print(test_data)

    # preprocessing the dataset
    test_data_scaled = sc.transform(test_data)

    # Obtaining X_test, y_test from the test data
    X_test = test_data_scaled[:, :][:, :-1]
    y_test = test_data_scaled[:, :][:, -1]
    y_test = np.array(tf.expand_dims(y_test, axis=-1))

    # reshaping the X_test for RNN
    X_test_scaled = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 3. Run prediction on the test data and output required plot and loss
    # predicting the output
    f_loss,f_mse = new_model.evaluate(X_test_scaled, y_test)
    print("The final test set loss is: ",f_loss)
    print("The final test set mse is: ",f_mse)
    y_test_predicted = new_model.predict(X_test_scaled)
    # to get the original value back
    plot_y_test_predicted = np.append(X_test, y_test_predicted, axis=1)

    plot_y_test_predicted = sc.inverse_transform(plot_y_test_predicted)
    test_data = pd.read_csv('.\\data\\test_data_RNN.csv')
    plot_y_test = test_data.iloc[:, -1].values

    plot_y_test_predicted = plot_y_test_predicted[:, :][:, -1]

    plt.plot(plot_y_test, color='red', label='real value')
    plt.plot(plot_y_test_predicted, color='blue', label='predicted price')
    plt.title('Model result comparision')
    plt.xlabel('Time')
    plt.ylabel('stock price')
    plt.legend()
    plt.show()

    # priting the MSE loss
    MSE = tf.keras.losses.MSE(plot_y_test, plot_y_test_predicted)
    print(MSE)

    # to better see the plot let's observe/plot a part of the dataset for better understanding
    plt.plot(plot_y_test[0:100], color='red', label='real value')
    plt.plot(plot_y_test_predicted[0:100],
             color='blue', label='predicted price')
    plt.title('model result comparision')
    plt.xlabel('Time')
    plt.ylabel('stock price')
    plt.legend()
    plt.show()
