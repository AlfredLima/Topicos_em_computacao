# -*- coding: utf-8 -*-
import benchmarks
import csv
import numpy
import time
import GWO as gwo
import matplotlib.pyplot as plt
import argparse
import sys

def runDiabetes(PopulationSize, Iterations, Dataset, Nodes, InputSize):
    dim = InputSize
    lbs = -1
    ubs = 1
    evolution = gwo.GWO(lbs, ubs, dim,
                        PopulationSize, Iterations, Dataset, Nodes)


def main():
    parser = argparse.ArgumentParser(
        description='GWO for optimizing neural networks')
    parser.add_argument('-p', '--population', default=12, type=int,
                        help='Population size')
    parser.add_argument('-it', '--iterations', type=int,
                        default=100,
                        help='Number of iterations')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset path', default="pima-indians-diabetes.csv")
    parser.add_argument('-n', '--nodes', type=int,
                        help='Number of hidden nodes in neural network', default=12)
    parser.add_argument('-i', '--input_size', type=int,
                        help='Input size', default=8)
    args = parser.parse_args(sys.argv[1:])

    print("**************************************")
    print("* GWO for optimizing neural networks *")
    print("**************************************")

    print(args.population)
    
    runDiabetes(PopulationSize=args.population,
                Iterations=args.iterations, Dataset=args.dataset, Nodes=args.nodes, InputSize = args.input_size)


if __name__ == "__main__":
    main()
