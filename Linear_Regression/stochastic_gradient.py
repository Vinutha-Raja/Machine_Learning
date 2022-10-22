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
    for i in range(len(data)):
        grad = gradient(vector, data.iloc[i].values)
        print(grad)
        print(learn_rate * grad)
        vector_new = vector + learn_rate * grad
        print("vector_new", vector_new)
        loss = loss_fucntion(vector_new, data)
        cost_function.append(loss)
        print("loss at {}: {}".format(i, loss))
        learn_rate = learn_rate/2
    return vector_new, cost_function


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

cost_function = []
training_data = read_data('./concrete/train.csv')
test_data = read_data('./concrete/test.csv')
initial_w = np.array([0] * 7, dtype=float)
v, cost_function = gradient_descent(start=initial_w, learn_rate=0.125, n_iter=1000, data=training_data, cost_function=cost_function)
print("final omega:", v)
print("cost function of test data:", loss_fucntion(v, test_data))
xpoints = [i for i in range(len(cost_function))]
ypoints = np.array(cost_function)
plt.plot(xpoints, ypoints)
plt.show()





