# -*- coding: utf-8 -*-
from GWO import GWO
import argparse
import sys
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from shapely.ops import cascaded_union
from test import run_pygame
from test import create_base_stations
from test import createPoints2Poly
from test import create_city
import math


centers_1 = [(219, 287), (219, 393), (312, 234), (312, 234),
             (312, 340), (312, 448), (405, 287), (405, 393)]

centers_2 = [(127,124), (127,231), (127,340), (127,448), (220,178), (220,286), (220,394), (220,502), (312,124), (312,231), (312,340), (312,448), (406,178), (406,286), (406,394), (406,502), (499,124), (499,231), (499,340), (499,448)]


centers_3 = [(127,231), (127,340), (127,448), (220,178), (220,286), (220,394), (312,124), (312,231), (406,178), (406,286), (406,394), (499,231), (499,340), (499,448)]

K = [7, 3, 6]

D = [63, 63, 63]

R = [63, 170, 126]

centers = [centers_1, centers_2, centers_3]


def main():
    parser = argparse.ArgumentParser(
        description='GWO for optimizing neural networks')
    parser.add_argument('-p', '--population', default=12, type=int,
                        help='Population size')
    parser.add_argument('-it', '--iterations', type=int,
                        default=100,
                        help='Number of iterations')
    parser.add_argument('-c', '--choice', type=int,
                        default=1,
                        help='Choice the instance')
    
    args = parser.parse_args(sys.argv[1:])

    print("**************************************")
    print("*  GWO for optimizing base stations  *")
    print("**************************************")


    if args.choice < 1 or args.choice > 3 :
        print("The instance choice doesn't exists")
        return 
    
    low_boundary = 0 + R[args.choice-1]
    
    upper_boundary = 625 - R[args.choice-1]
    
    city = create_city(centers[args.choice-1], D[args.choice-1])
    
    center_base_stations = GWO(low_boundary, upper_boundary, (K[args.choice-1], 2), args.population, args.iterations, R[args.choice-1], D[args.choice-1], city, centers[args.choice-1])

if __name__ == "__main__":
    main()
