import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    dataset = []
    for r in data:
        f = []
        for x in r:
            f.append(float(x))
        dataset.append(f)
    return dataset


def calculate_norm(w):
    norm_val = 0
    for i in w:
        norm_val += math.pow(i, 2)
    norm_val = math.sqrt(norm_val)
    return norm_val


def vector_prod(vec1, vec2):
    prod = 0
    for i in range(len(vec1)):
        prod += vec1[i] * vec2[i]
    return prod


def gradient_descent(start, learn_rate, data):
    cost_function = []
    weight = start
    norm_d = 1
    cost_function.append(loss_fucntion(weight, data))
    while norm_d > 1e-6:
        grad_v = []
        for i in range(len(weight)):
            s = 0
            for j in range(len(data)):
                s -= (data[j][-1] - vector_prod(weight, data[j])) * data[j][i]
            grad_v.append(s / len(data))
        next_weight = []
        for i in range(len(weight)):
            next_weight.append(weight[i] - learn_rate * grad_v[i])
        weight_diff = []
        for i in range(len(weight)):
            weight_diff.append(next_weight[i] - weight[i])
        norm_d = calculate_norm(weight_diff)
        weight = next_weight
        cost_function.append(loss_fucntion(weight, data))

    return weight, cost_function


def loss_fucntion(w_vector, data):
    summation = 0
    for i in data:
        prod = vector_prod(w_vector, i)
        s = 0.5 * math.pow(i[-1] - prod, 2)
        summation += s
    return summation


training_data = read_data('./concrete/train.csv')
test_data = read_data('./concrete/test.csv')
initial_w = np.array([0] * 7)
v, cost_function0 = gradient_descent(start=initial_w, learn_rate=0.5,  data=training_data)
v1, cost_function1 = gradient_descent(start=initial_w, learn_rate=0.25,  data=training_data)
v2, cost_function2 = gradient_descent(start=initial_w, learn_rate=0.125,  data=training_data)
v3, cost_function3 = gradient_descent(start=initial_w, learn_rate=0.0625,  data=training_data)
v4, cost_function4 = gradient_descent(start=initial_w, learn_rate=0.03125,  data=training_data)

print("final weight vector for learning rate 0.5 is", v)
print("cost function of test data:", loss_fucntion(v, test_data))

print("final weight vector for learning rate 0.25 is", v1)
print("cost function of test data:", loss_fucntion(v1, test_data))

print("final weight vector for learning rate 0.125 is", v2)
print("cost function of test data:", loss_fucntion(v2, test_data))

print("final weight vector for learning rate 0.0625 is", v3)
print("cost function of test data:", loss_fucntion(v3, test_data))

print("final weight vector for learning rate 0.03125 is", v4)
print("cost function of test data:", loss_fucntion(v4, test_data))

plt.plot(cost_function0, label='r=0.5')
plt.plot(cost_function1, label='r=0.25')
plt.plot(cost_function2, label='r=0.125')
plt.plot(cost_function3, label='r=0.0625')
plt.plot(cost_function4, label='r=0.03125')
plt.legend()
plt.title("Batch Gradient Descent")
plt.show()

#
# final weight vector for learning rate 0.5 is [0.9211995194368011, 0.807933803591768, 0.8735844368405145, 1.314007558321237, 0.13386551283048725, 1.5985765136683072, 1.019934869939579]
# cost function of test data: 23.360817352947507
# final weight vector for learning rate 0.25 is [0.9208493516424054, 0.807573106777766, 0.8731943045234826, 1.313727240883312, 0.13380724660358637, 1.5981054611331282, 1.0195775968876037]
# cost function of test data: 23.360310175924564
# final weight vector for learning rate 0.125 is [0.9201482947214039, 0.8068509701281756, 0.8724132362318268, 1.3131660285643048, 0.13369059412359444, 1.5971623857126376, 1.0188623148148859]
# cost function of test data: 23.359295350661025
# final weight vector for learning rate 0.0625 is [0.9187465184994094, 0.8054070446007301, 0.8708514758009726, 1.312043874198934, 0.13345734534194673, 1.5952766890446641, 1.0174320951401092]
# cost function of test data: 23.357268482917085
# final weight vector for learning rate 0.03125 is [0.9159435089423117, 0.8025197527565232, 0.8677285597856376, 1.3098000000617356, 0.1329909381124036, 1.5915060260107334, 1.0145722096931686]
# cost function of test data: 23.35322470493231
#

# Part c - Analytical solution
X = np.transpose(np.array([row[:-1] for row in training_data]))
Y = np.array([row[-1] for row in training_data])

weight_vector = np.dot(np.linalg.inv(np.dot(X, np.transpose(X))), np.dot(X, Y))

print("Optimal weight vector: {}".format(weight_vector))