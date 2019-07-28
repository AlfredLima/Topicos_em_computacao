import random
import numpy
import math
from solution import solution
import time
# import Plot


def GWO(obj_func, lower_bounds, upper_bounds, dim, wolf_population, max_iter, min_or_max=0):
    # Min or Max -> 0 is minimo and 1 is maximo
    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")
    if min_or_max:
        Alpha_score = -Alpha_score

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")
    if min_or_max:
        Beta_score = -Beta_score

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")
    if min_or_max:
        Delta_score = -Delta_score

    if not isinstance(lower_bounds, list):
        lower_bounds = [lower_bounds] * dim
    if not isinstance(upper_bounds, list):
        upper_bounds = [upper_bounds] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((wolf_population, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(
            0, 1, wolf_population) * (upper_bounds[i] - lower_bounds[i]) + lower_bounds[i]

    Convergence_curve = numpy.zeros(max_iter)
    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, max_iter):
        for i in range(0, wolf_population):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(
                    Positions[i, j], lower_bounds[j], upper_bounds[j])

            # Calculate objective function for each search agent
            fitness = obj_func(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if min_or_max:
                if fitness > Alpha_score:
                    Alpha_score = fitness  # Update alpha
                    Alpha_pos = Positions[i, :].copy()

                if (fitness < Alpha_score and fitness > Beta_score):
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()

                if (fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score):
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()
            else:
                if fitness < Alpha_score:
                    Alpha_score = fitness  # Update alpha
                    Alpha_pos = Positions[i, :].copy()

                if (fitness > Alpha_score and fitness < Beta_score):
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()

                if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()

        a = 2-l*((2)/max_iter)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, wolf_population):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2*a*r1-a  # Equation (3.3)
                C1 = 2*r2  # Equation (3.4)

                # Equation (3.5)-part 1
                D_alpha = abs(C1*Alpha_pos[j]-Positions[i, j])
                X1 = Alpha_pos[j]-A1*D_alpha  # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2*a*r1-a  # Equation (3.3)
                C2 = 2*r2  # Equation (3.4)

                # Equation (3.5)-part 2
                D_beta = abs(C2*Beta_pos[j]-Positions[i, j])
                X2 = Beta_pos[j]-A2*D_beta  # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2*a*r1-a  # Equation (3.3)
                C3 = 2*r2  # Equation (3.4)

                # Equation (3.5)-part 3
                D_delta = abs(C3*Delta_pos[j]-Positions[i, j])
                X3 = Delta_pos[j]-A3*D_delta  # Equation (3.5)-part 3

                Positions[i, j] = (X1+X2+X3)/3  # Equation (3.7)

        Convergence_curve[l] = Alpha_score

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd-timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.obj_funcname = obj_func.__name__

    return s
