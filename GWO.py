import random
import numpy
import math
from solution import solution
import time

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


def GWO(lower_bounds, upper_bounds, dim, wolf_population, max_iter):
    print("Run...")
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    inputs = dataset[:, 0:dim]
    output = dataset[:, dim]
    # model
    model = Sequential()
    model.add(Dense(12, input_dim=dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Min or Max -> 0 is minimo and 1 is maximo
    # initialize alpha, beta, and delta_pos

    dim_layers = 12 * 8 + 12 * 1

    Alpha_pos = numpy.zeros(dim_layers)
    Alpha_score = -float("inf")

    Beta_pos = numpy.zeros(dim_layers)
    Beta_score = -float("inf")

    Delta_pos = numpy.zeros(dim_layers)
    Delta_score = -float("inf")

    if not isinstance(lower_bounds, list):
        lower_bounds = [lower_bounds] * dim_layers
    if not isinstance(upper_bounds, list):
        upper_bounds = [upper_bounds] * dim_layers

    # Initialize the positions of search agents
    Positions = numpy.zeros((wolf_population, dim_layers))
    for i in range(dim_layers):
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
            for j in range(dim_layers):
                Positions[i, j] = numpy.clip(
                    Positions[i, j], lower_bounds[j], upper_bounds[j])

            # Calculate objective function for each search agent
            cpy = Positions[i, :].copy()
            first_layer = []

            w = numpy.zeros((8, 12))
            for k in range(8):
                for m in range(12):
                    w[k][m] = cpy[12*k + m]
            first_layer.append(w)
            w = numpy.zeros(12)
            first_layer.append(w)

            second_layer = []
            w = numpy.zeros((12, 1))
            for k in range(12):
                w[k] = cpy[8*12 + k]
            second_layer.append(w)
            w = numpy.zeros(1)
            second_layer.append(w)

            model.layers[0].set_weights(first_layer)
            model.layers[1].set_weights(second_layer)

            _, fitness = model.evaluate(inputs, output, verbose=0)
            fitness = fitness*100

            # Update Alpha, Beta, and Delta
            if fitness > Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness < Alpha_score and fitness > Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2-l*((2)/max_iter)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, wolf_population):
            for j in range(0, dim_layers):

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

        print("Run", l, ": Accuary is", Alpha_score, "%")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.convergence = Convergence_curve
    s.optimizer = "GWO"

    return s
