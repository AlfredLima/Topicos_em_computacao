# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
import benchmarks
import csv
import numpy
import time
import GWO as gwo
import matplotlib.pyplot as plt
# from numpy import loadtxt
# from keras.models import Sequential
# from keras.layers import Dense

# # load the dataset
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# # split into input (X) and output (y) variables
# X = dataset[:, 0:8]
# y = dataset[:, 8]

# # define the keras model
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # compile the keras model
# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

# # fit the keras model on the dataset
# model.fit(X, y, epochs=150, batch_size=10)

# # evaluate the keras model
# _, accuracy = model.evaluate(X, y)
# print('Accuracy: %.2f' % (accuracy*100))


def function_F1(vector):
    return sum(vector)


# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 12
Iterations = 50

dim = 5
lbs = [0.5]*dim
ubs = [1]*dim

output = gwo.GWO(function_F1, lbs, ubs, dim, PopulationSize, Iterations, 0)

plt.plot(output.convergence, 'ro')
plt.ylabel('some numbers')
plt.show()
