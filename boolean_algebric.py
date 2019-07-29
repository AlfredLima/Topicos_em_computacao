# -*- coding: utf-8 -*-
import csv
import numpy
import time
import matplotlib.pyplot as plt
import argparse
import sys
import random
import itertools

def runMakeCircuit(InputSize):
    with open('digitalCircuit.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)

        a = [0,1]
        combinations = None
        combinations = list(itertools.product(a,a))
        if InputSize == 1:
            combinations = list(itertools.product(a))
        elif InputSize == 2:
            combinations = list(itertools.product(a,a))
        elif InputSize == 3:
            combinations = list(itertools.product(a,a,a))
        elif InputSize == 4:
            combinations = list(itertools.product(a,a,a,a))
        elif InputSize == 5:
            combinations = list(itertools.product(a,a,a,a,a))
        elif InputSize == 6:
            combinations = list(itertools.product(a,a,a,a,a,a))
        elif InputSize == 7:
            combinations = list(itertools.product(a,a,a,a,a,a,a))
        elif InputSize == 8:
            combinations = list(itertools.product(a,a,a,a,a,a,a,a))
        else:
            print("Erro only in [1,8]")
            return

        for c in combinations:
            line = list(c) + [random.randint(0, 1)]
            line = (list(map(str, line)))
            writer.writerow(line)

        writeFile.close()

def main():
    parser = argparse.ArgumentParser(
        description='Generate random digital circuit')
    parser.add_argument('-i', '--input_size', type=int,
                        help='Input size', default=2)
    args = parser.parse_args(sys.argv[1:])

    print("**************************************")
    print("*  Generate random digital circuit   *")
    print("**************************************")
    
    runMakeCircuit(InputSize = args.input_size)


if __name__ == "__main__":
    main()