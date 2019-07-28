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
    lbs = -1
    ubs = 1
    evolution = gwo.GWO(lbs, ubs, dim,
                        PopulationSize, Iterations)


runDiabetes(100, 100)
