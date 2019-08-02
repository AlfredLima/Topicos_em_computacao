# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""

import random
import numpy as np
import math
import time
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from shapely.ops import cascaded_union
from test import create_base_stations
from test import create_city
from test import create_base_stations_points
from test import create_city_points
from test import run_pygame

def calculate_fitness(wolf, radius, city) :
    circles = []
    
    for center in wolf :
        point = Point(center[0], center[1])
        circles.append(point.buffer(radius))

    base_stations_polygon = cascaded_union(circles)
    intersection_area = base_stations_polygon.intersection(city).area
    
    return base_stations_polygon.intersection(city).area / city.area


def GWO(low_boundary,upper_boundary,dimension,population,iterations,radius, D, city, city_centers):


    #iterations=1000
    #low_boundary=-100
    #upper_boundary=100
    #dimension=30
    #population=5

    print("low boundary", low_boundary)
    print("upper boundary", upper_boundary)
    print("dimension", dimension)
    print("population", population)
    print("iterations", iterations)

    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dimension)
    Alpha_score=-float("inf")

    Beta_pos=np.zeros(dimension)
    Beta_score=-float("inf")

    Delta_pos=np.zeros(dimension)
    Delta_score=-float("inf")

    #Initialize the positions of search agents
    Positions = []
    for i in range(population) :
        Positions.append(np.zeros(dimension))

    for i in range(population):
        Positions[i][:, 0] = np.random.uniform(0,1, dimension[0]) * (upper_boundary - low_boundary) + low_boundary
        Positions[i][:, 1] = np.random.uniform(0,1, dimension[0]) * (upper_boundary - low_boundary) + low_boundary
        

    Convergence_curve=np.zeros(iterations)

     # Loop counter
    print("GWO is optimizing")

    # Main loop
    for l in range(0,iterations):
        for i in range(0,population):

            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i]=np.clip(Positions[i], low_boundary, upper_boundary)

            # Calculate objective function for each search agent
            fitness = calculate_fitness(Positions[i], radius, city)

            # Update Alpha, Beta, and Delta
            if fitness > Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i].copy()


            if (fitness < Alpha_score and fitness > Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i].copy()


            if (fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score):
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i].copy()


        a=2-l*((2)/iterations); # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0,population):
            for j in range(0,dimension[0]) :
                for k in range(0, dimension[1]) :

                    r1=random.random() # r1 is a random number in [0,1]
                    r2=random.random() # r2 is a random number in [0,1]

                    A1=2*a*r1-a; # Equation (3.3)
                    C1=2*r2; # Equation (3.4)

                    D_alpha=abs(C1*Alpha_pos[j][k]-Positions[i][j][k]); # Equation (3.5)-part 1
                    X1=Alpha_pos[j][k]-A1*D_alpha; # Equation (3.6)-part 1

                    r1=random.random()
                    r2=random.random()

                    A2=2*a*r1-a; # Equation (3.3)
                    C2=2*r2; # Equation (3.4)

                    D_beta=abs(C2*Beta_pos[j][k]-Positions[i][j][k]); # Equation (3.5)-part 2
                    X2=Beta_pos[j][k]-A2*D_beta; # Equation (3.6)-part 2

                    r1=random.random()
                    r2=random.random()

                    A3=2*a*r1-a; # Equation (3.3)
                    C3=2*r2; # Equation (3.4)

                    D_delta=abs(C3*Delta_pos[j][k]-Positions[i][j][k]); # Equation (3.5)-part 3
                    X3=Delta_pos[j][k]-A3*D_delta; # Equation (3.5)-part 3

                    Positions[i][j][k] = (X1+X2+X3)/3  # Equation (3.7)




        Convergence_curve[l]=Alpha_score;

        if (l%1==0):
               print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(Alpha_score)]);
               run_pygame(create_city_points(city_centers, D), create_base_stations_points(Alpha_pos, radius))

        


    return Alpha_pos
