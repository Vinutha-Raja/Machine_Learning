import numpy as np
import pandas as pd


def gaussian_prediction_func(gamma, sol, x, x1, y):
    kernel = gaussian_kernel_func(x, x1, gamma)
    sol_y = np.reshape(np.multiply(sol, np.reshape(y, (-1, 1))), (1, -1))
    pred = np.matmul(sol_y, kernel)
    pred = np.reshape(pred, (-1, 1))
    return pred


def gaussian_kernel_func(x1, x2, gamma):
    x_i = np.tile(x1, (1, x2.shape[0]))
    x_i = np.reshape(x_i, (-1, x1.shape[1]))
    x_j = np.tile(x2, (x1.shape[0], 1))
    k_xi_xj = np.exp(np.sum(np.square(x_i - x_j), axis=1) / -gamma)
    k_xi_xj = np.reshape(k_xi_xj, (x1.shape[0], x2.shape[0]))
    return k_xi_xj


def kernel_perceptron(x, y, gamma):
    num_s = x.shape[0]
    index = np.arange(num_s)
    weight_vec = np.array([x for x in range(num_s)])
    weight_vec = np.reshape(weight_vec, (-1, 1))
    y = np.reshape(y, (-1, 1))
    kernel = gaussian_kernel_func(x, x, gamma)
    for t in range(100):
        np.random.shuffle(index)
        for i in range(num_s):
            weight_y = np.multiply(weight_vec, y)
            weight_y_kernel = np.matmul(kernel[index[i], :], weight_y)
            # uodate weight only if incorrect pred
            if weight_y_kernel * y[index[i]] <= 0:
                weight_vec[index[i]] = weight_vec[index[i]] + 1
    return weight_vec


train = pd.read_csv('./bank-note/train.csv', header=None)
# r = train.shape[0]
c = train.shape[1]
train_vals = train.values
train_data = np.copy(train)
train_data[:, c - 1] = 1
train_y = train_vals[:, c - 1]
train_y = 2 * train_y - 1  # converting 0,1 to 1, -1
# print(np.reshape/(train_y, (-1, 1)))
test = pd.read_csv('./bank-note/test.csv', header=None)
# r = train.shape[0]
c = test.shape[1]
test_vals = test.values
test_data = np.copy(test)
test_data[:, c - 1] = 1
test_y = test_vals[:, c - 1]
test_y = 2 * test_y - 1  # converting 0,1 to 1, -1

gamma_val = np.array([0.1, 0.5, 1, 5, 100])
print('Kernel Perceptron Algorithm')
for g in gamma_val:
    sol = kernel_perceptron(train_data, train_y, g)
    prediction_Array = gaussian_prediction_func(g, sol, train_data, train_data, train_y)
    prediction_Array[prediction_Array > 0] = 1
    prediction_Array[prediction_Array <= 0] = -1
    # print(prediction_Array)
    train_y1 = np.reshape(train_y, (-1, 1))
    train_err_count = np.sum(prediction_Array != train_y1)

    test_prediction = gaussian_prediction_func(g, sol, train_data, test_data, train_y)
    test_prediction[test_prediction > 0] = 1
    test_prediction[test_prediction <= 0] = -1
    # print(prediction_Array)
    test_y1 = np.reshape(test_y, (-1, 1))
    test_err_count = np.sum(test_prediction != test_y1)

    print('Gamma: {}, train_err_count:{}, test_err_count: {} '.format(g, train_err_count/len(train_y), test_err_count/len(test_y)))

# Gamma: 0.1, train_err_count:0.0, test_err_count: 0.002
# Gamma: 0.5, train_err_count:0.0, test_err_count: 0.002
# Gamma: 1.0, train_err_count:0.0, test_err_count: 0.002
# Gamma: 5.0, train_err_count:0.008027522935779817, test_err_count: 0.006
# Gamma: 100.0, train_err_count:0.15711009174311927, test_err_count: 0.17