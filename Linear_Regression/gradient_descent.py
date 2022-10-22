import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    df = pd.read_csv(file_name, header=None,
                     names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label'])
    print(df)
    return df


def gradient_descent(start, learn_rate, n_iter, data, cost_function):
    vector = start
    inital_loss = loss_fucntion(vector, data)
    print("loss initial:", inital_loss)
    cost_function.append(inital_loss)
    for i in range(n_iter):
        grad = gradient(vector, data)
        vector_new = vector - learn_rate * grad
        norm = np.linalg.norm(np.subtract(vector_new, vector))
        vector = vector_new
        if norm < 0.000001:
            print("converging")
            break
        loss = loss_fucntion(vector, data)
        cost_function.append(loss)
        print("loss at {}: {}".format(i, loss))
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
    grad = np.array([0] * 7)
    g_0, g_1, g_2, g_3, g_4, g_5, g_6 = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(data.index)):
        x_i = np.array([data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2],
                       data.iloc[i, 3], data.iloc[i, 4], data.iloc[i, 5], data.iloc[i, 6]])
        wt_xi = np.matmul(np.transpose(w_vector), x_i)
        yi_wtxi = data.iloc[i, 7] - wt_xi
        g_0 += yi_wtxi * x_i[0]
        g_1 += yi_wtxi * x_i[1]
        g_2 += yi_wtxi * x_i[2]
        g_3 += yi_wtxi * x_i[3]
        g_4 += yi_wtxi * x_i[4]
        g_5 += yi_wtxi * x_i[5]
        g_6 += yi_wtxi * x_i[6]
    grad[0] = - 1 * g_0
    grad[1] = - 1 * g_1
    grad[2] = - 1 * g_2
    grad[3] = - 1 * g_3
    grad[4] = - 1 * g_4
    grad[5] = - 1 * g_5
    grad[6] = - 1 * g_6
    return grad

cost_function = []
training_data = read_data('./concrete/train.csv')
test_data = read_data('./concrete/test.csv')
initial_w = np.array([0] * 7)
v, cost_function = gradient_descent(start=initial_w, learn_rate=0.001, n_iter=1000, data=training_data, cost_function=cost_function)
print("final omega:", v)
print("cost function of test data:", loss_fucntion(v, test_data))
xpoints = [i for i in range(len(cost_function))]
ypoints = np.array(cost_function)
plt.plot(xpoints, ypoints)
plt.show()








