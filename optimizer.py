# -*- coding: utf-8 -*-
import benchmarks
import csv
import numpy
import time
import GWO as gwo
import matplotlib.pyplot as plt
import argparse
import sys

def runDiabetes(PopulationSize, Iterations, Dataset):
    dim = 8
    lbs = -1
    ubs = 1
    evolution = gwo.GWO(lbs, ubs, dim,
                        PopulationSize, Iterations, Dataset)


def main() :
    parser = argparse.ArgumentParser(description='GWO for optimizing neural networks')
    parser.add_argument('-p', '--population', default=12, type=int,
                        help='Population size')
    parser.add_argument('-it', '--iterations', type=int,
                        default=100,
                        help='Number of iterations')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset path', default="pima-indians-diabetes.csv")
    args = parser.parse_args(sys.argv[1:])

    print("**************************************")
    print("* GWO for optimizing neural networks *")
    print("**************************************")

    print(args.population)
    
    runDiabetes(PopulationSize=args.population, Iterations=args.iterations, Dataset=args.dataset)



if __name__ == "__main__":
    main()
