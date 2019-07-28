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


def runDiabetes(PopulationSize, Iterations):
    dim = 8
    lbs = [-1]*dim
    ubs = [1]*dim
    # load the dataset

    evolution = gwo.GWO(lbs, ubs, dim,
                        PopulationSize, Iterations, 0)
# # evaluate the keras model
# _, accuracy = model.evaluate(X, y)
# print('Accuracy: %.2f' % (accuracy*100))
