import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# def read_data(file_name):
#     df = pd.read_csv(file_name, header=None,
#                      names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'label'])
#     # print(df)
#     return df


def primal_svm(train_data_x, train_y, C, lr, a):
    r = train_data_x.shape[0]
    c = train_data_x.shape[1]
    weight_vec = np.zeros(c)
    axis = np.arange(r)

    for t in range(100):
        np.random.shuffle(axis)
        x = train_data_x[axis, :]
        # print("x", len(x))
        y = train_y[axis]
        # print("y", len(y))
        for j in range(r):
            y_wt_x = y[j] * np.sum(np.multiply(weight_vec, x[j, :]))
            w = np.copy(weight_vec)
            w[c - 1] = 0
            if y_wt_x <= 1:
                w = w - C * r * y[j] * x[j, :]
            lr = lr / (1 + lr / a * t)
            weight_vec = weight_vec - lr * w

    return weight_vec


def primal_svm_1(train_data_x, train_y, C, lr, a):
    r = train_data_x.shape[0]
    c = train_data_x.shape[1]
    weight_vec = np.zeros(c)
    axis = np.arange(r)

    for t in range(100):
        np.random.shuffle(axis)
        x = train_data_x[axis, :]
        # print("x", len(x))
        y = train_y[axis]
        # print("y", len(y))
        for j in range(train_data_x.shape[0]):
            y_wt_x = y[j] * np.sum(np.multiply(weight_vec, x[j, :]))
            w = np.copy(weight_vec)
            w[c - 1] = 0
            if y_wt_x <= 1:
                w = w - C * r * y[j] * x[j, :]
            lr = lr / (1 + t)
            weight_vec = weight_vec - lr * w

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
print("Primal SVM 2(a)")
C_vals = [100/873, 500/873, 700/873]
for c in C_vals:
    # 2A
    # print(train_data)
    w = primal_svm(train_data, train_y, c, 0.1, 0.1)
    w1 = np.reshape(w, (5, 1))
    prediction_Array = np.matmul(train_data, w1)
    # print(prediction_Array)
    prediction_Array[prediction_Array > 0] = 1
    prediction_Array[prediction_Array <= 0] = -1
    # print(prediction_Array)
    train_y1 = np.reshape(train_y, (-1, 1))
    train_err_count = np.sum(prediction_Array != train_y1)
    # print(train_err_count)
    test_prediction = np.matmul(test_data, w1)
    test_prediction[test_prediction > 0] = 1
    test_prediction[test_prediction <= 0] = -1
    # print(prediction_Array)
    test_y1 = np.reshape(test_y, (-1, 1))
    test_err_count = np.sum(test_prediction != test_y1)

    # print("Primal SVM 2(a)=> C: {}, weight_vector: {}, train_err_count: {}, test_err_count: {}".format(c, w, train_err_count/len(train_y), test_err_count/len(test_y)))
    print("C: {}, weight_vector: {} ".format(c, w, ))
    print(
        "train_err_count: {}, test_err_count: {}".format(train_err_count / len(train_y), test_err_count / len(test_y)))


print("Primal SVM 2(b)")
for c in C_vals:
    # 2B
    w = primal_svm_1(train_data, train_y, c, 0.1, 0.1)
    w1 = np.reshape(w, (5, 1))
    prediction_Array = np.matmul(train_data, w1)
    # print(prediction_Array)
    prediction_Array[prediction_Array > 0] = 1
    prediction_Array[prediction_Array <= 0] = -1
    # print(prediction_Array)
    train_y1 = np.reshape(train_y, (-1, 1))
    train_err_count = np.sum(prediction_Array != train_y1)
    # print(train_err_count)
    test_prediction = np.matmul(test_data, w1)
    # print(test_prediction)
    test_prediction[test_prediction > 0] = 1
    test_prediction[test_prediction <= 0] = -1
    # print(prediction_Array)
    test_y1 = np.reshape(test_y, (-1, 1))
    test_err_count = np.sum(test_prediction != test_y1)

    print("C: {}, weight_vector: {} ".format(c, w,))
    print("train_err_count: {}, test_err_count: {}".format(train_err_count/len(train_y), test_err_count/len(test_y)))


# 5th question theory part
def primal_svm_theory(train_data_x, train_y, C, lr, a):
    r = train_data_x.shape[0]
    c = train_data_x.shape[1]
    weight_vec = np.array([0] * c, dtype=float)
    axis = np.arange(r)

    for t in range(1):
        # np.random.shuffle(axis)
        x = train_data_x[axis, :]
        # print("x", len(x))
        y = train_y[axis]
        # print("y", len(y))
        for j in range(train_data_x.shape[0]):
            y_wt_x = y[j] * np.sum(np.multiply(weight_vec, x[j, :]))
            w = np.copy(weight_vec)
            w[c - 1] = 0
            if y_wt_x <= 1:
                w = w - C * r * y[j] * x[j, :]
            print("lr: {}, weight: {}".format(lr, weight_vec))
            lr = lr / 2
            weight_vec = weight_vec - lr * w

    return weight_vec


train_data_x = np.array([[0.5, -1, 0.3, 1], [-1, -2, -2, 1], [1.5, 0.2, -2.5, 1]])
train_y = np.array([1, -1, 1])
primal_svm_theory(train_data_x, train_y, 1/3, 0.01, 0)
