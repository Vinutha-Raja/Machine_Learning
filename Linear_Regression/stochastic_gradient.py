import math
import pandas as pd
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
            r = random.randint(0, len(data.index) - 1)
            grad = gradient(vector, data.iloc[r].values)
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
        x_i = np.array([data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2],
                       data.iloc[i, 3], data.iloc[i, 4], data.iloc[i, 5],  data.iloc[i, 6]])
        wt_xi = np.matmul(np.transpose(w_vector), x_i)
        s = math.pow(data.iloc[i, 7] - wt_xi, 2)
        summation += s

    loss = summation / 2
    return loss


def gradient(w_vector, data):
    grad = np.array([0] * 7, dtype=float)

    x_i = np.array(data[0:7])

    wt_xi = np.matmul(np.transpose(w_vector), x_i)
    yi_wtxi = data[7] - wt_xi
    grad[0] = yi_wtxi * x_i[0]
    grad[1] = yi_wtxi * x_i[1]
    grad[2] = yi_wtxi * x_i[2]
    grad[3] = yi_wtxi * x_i[3]
    grad[4] = yi_wtxi * x_i[4]
    grad[5] = yi_wtxi * x_i[5]
    grad[6] = yi_wtxi * x_i[6]

    return grad

training_data = read_data('./concrete/train.csv')
test_data = read_data('./concrete/test.csv')
initial_w = np.array([0] * 7, dtype=float)
v, cost_function0 = gradient_descent(start=initial_w, learn_rate=0.01, n_iter=100, data=training_data)
v1, cost_function1 = gradient_descent(start=initial_w, learn_rate=0.005,   n_iter=100, data=training_data)
v2, cost_function2 = gradient_descent(start=initial_w, learn_rate=0.0025,   n_iter=100, data=training_data)
v3, cost_function3 = gradient_descent(start=initial_w, learn_rate=0.00125,  n_iter=100,  data=training_data)

print("final weight vector for learning rate 0.01 is", v)
print("cost function of test data:", loss_fucntion(v, test_data))

print("final weight vector for learning rate 0.005 is", v1)
print("cost function of test data:", loss_fucntion(v1, test_data))

print("final weight vector for learning rate 0.0025 is", v2)
print("cost function of test data:", loss_fucntion(v2, test_data))

print("final weight vector for learning rate 0.00125 is", v3)
print("cost function of test data:", loss_fucntion(v3, test_data))

ypoints = np.array(cost_function0)
y1points = np.array(cost_function1)
y2points = np.array(cost_function2)
y3points = np.array(cost_function3)

plt.plot(ypoints, label='r=0.5')
plt.plot(y1points, label='r=0.25')
plt.plot(y2points, label='r=0.125')
plt.plot(y3points, label='r=0.0625')
plt.legend()
plt.title("Stochastic Gradient Descent")
plt.show()




# final weight vector for learning rate 0.01 is [0.56403871 0.5048317  0.60967499 0.99997241 0.21442537 1.09029873
#  0.70795981]
# cost function of test data: 23.21366165657389
# final weight vector for learning rate 0.005 is [0.30052925 0.24597909 0.24727974 0.85409401 0.04567808 0.80205297
#  0.40706346]
# cost function of test data: 23.147648295779675
# final weight vector for learning rate 0.0025 is [ 0.10071544  0.04694021 -0.04813319  0.70939516 -0.02108462  0.59083173
#   0.21150531]
# cost function of test data: 24.410207063793127
# final weight vector for learning rate 0.00125 is [-0.02594705 -0.19636083 -0.17964765  0.37519458 -0.02329348  0.11234792
#   0.01811732]
# cost function of test data: 21.00907269287997

#
# final weight vector for learning rate 0.01 is [ 0.00816528 -0.15816688 -0.21736911  0.5725893   0.02800408  0.41484012 0.10191847]
# cost function of test data: 24.224399129934483
# final weight vector for learning rate 0.005 is [-0.06090565 -0.17273488 -0.22669199  0.5341542   0.02777341  0.37186438 -0.022764  ]
# cost function of test data: 25.988458342462824
# final weight vector for learning rate 0.0025 is [-0.08045259 -0.19515645 -0.23677321  0.51051759 -0.04998005  0.24454925 0.01765338]
# cost function of test data: 22.37194521495275
# final weight vector for learning rate 0.00125 is [-0.08059784 -0.22052829 -0.20829198  0.50144058  0.00836205  0.21612368 -0.04221624]
# cost function of test data: 22.881391845250235


