import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.transpose([[1, -1, 2], [1, 1, 3], [-1, 1, 0], [1, 2, -4], [3, -1, -1]])
Y = np.array([1, 4, -1, -2, 0])

weight_vector = np.dot(np.linalg.inv(np.dot(X, np.transpose(X))), np.dot(X, Y))

print("Optimal weight vector is: {}".format(weight_vector))

import math
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def read_data(file_name):
    df = pd.read_csv(file_name, header=None,
                     names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label'])
    # print(df)
    return df


def gradient_descent(start, learn_rate, n_iter, data):
    cost_function = []
    vector = start
    inital_loss = loss_fucntion(vector, data)
    # print("loss initial:", inital_loss)
    cost_function.append(inital_loss)
    prev_loss = inital_loss
    diff = 1
    for j in range(n_iter):
        if diff <= 1e-6:
            break
        for i in range(len(data)):
            print("step: ", i)
            print("weight:", vector)

            r = random.randint(0, len(data.index) - 1)
            grad = gradient(vector, data.iloc[r].values)
            print("gradient:", grad)
            # print(grad)
            # print(learn_rate * grad)
            vector_new = vector + learn_rate * grad
            vector = vector_new
            # print("vector_new", vector_new)
            loss = loss_fucntion(vector_new, data)
            cost_function.append(loss)
            # print("loss at {}: {}".format(i, loss))
            # learn_rate = learn_rate / 2
            diff = abs(loss - prev_loss)
            prev_loss = loss
    return vector, cost_function


def loss_fucntion(w_vector, data):
    summation = 0
    for i in range(len(data.index)):
        x_i = np.array([data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2]])
        wt_xi = np.matmul(np.transpose(w_vector), x_i)
        s = math.pow(data.iloc[i, 3] - wt_xi, 2)
        summation += s

    loss = summation / 2
    return loss


def gradient(w_vector, data):
    grad = np.array([0] * 3, dtype=float)

    x_i = np.array(data[0:3])

    wt_xi = np.matmul(np.transpose(w_vector), x_i)
    yi_wtxi = data[3] - wt_xi
    grad[0] = yi_wtxi * x_i[0]
    grad[1] = yi_wtxi * x_i[1]
    grad[2] = yi_wtxi * x_i[2]


    return grad

# training_data = read_data('./concrete/train.csv')
training_data = pd.DataFrame([[1, -1, 2, 1], [1, 1, 3, 4], [-1, 1, 0, -1], [1, 2, -4, -2], [3, -1, -1, 0]])
# test_data = read_data('./concrete/test.csv')
initial_w = np.array([0] * 3, dtype=float)
v, cost_function0 = gradient_descent(start=initial_w, learn_rate=0.1, n_iter=1, data=training_data)
