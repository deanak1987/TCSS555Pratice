# ML in Python, homework 3
# Name: Dean Kelley
# Date: December 1, 2022,
# Professor: Martine De Cock
# description: Neural network for predicting personality of Facebook users

import keras_tuner
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Loading the data
# There are 9500 users (rows)
# There are 81 columns for the LIWC features followed by columns for
# openness, conscientiousness, extraversion, agreeableness, neuroticism
# As the target variable, we select the extraversion column (column 83)
dataset = np.loadtxt("Facebook-User-LIWC-personality-HW3.csv", delimiter=",")
X = dataset[:,0:81]
y = dataset[:,83]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

# Training and testing a linear regression model
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
regression_score = metrics.mean_squared_error(y_test, y_pred)
print('MSE with linear regression:', metrics.mean_squared_error(y_test, y_pred))

# Training and testing a neural network
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=500)


def build_model(hp):
    model = Sequential()
    model.add(Dense(
        hp.Choice('units', [4, 8, 32]),
        activation='relu'))
    model.add(keras.layers.Dense(1, input_dim=81, activation='relu'))
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
    model.compile(
        # optimizer='adam',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mse'])
    return model


tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10)
tuner.search(X_train, y_train, epochs=500, validation_data=(X_val, y_val))
best_model = tuner.get_best_models()[0]
y_pred = best_model.predict(X_test)

nn_score = metrics.mean_squared_error(y_test, y_pred)
print('MSE with linear regression:', regression_score)
print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))
print(regression_score - nn_score)
