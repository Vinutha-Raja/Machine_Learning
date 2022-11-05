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
        l = f[4]
        f[4] = 1
        f.append(l)
        dataset.append(f)
    return dataset


def vector_prod(vec1, vec2):
    prod = 0
    for i in range(len(vec1)):
        prod += vec1[i] * vec2[i]
    return prod


def perceptron_algorithm(epoch, data, learning_rate):
    weight = [0, 0, 0, 0, 0]
    for i in range(epoch):
        np_data = np.array(data)
        np.random.shuffle(np_data)
        for row_num in range(len(np_data)):
            y_i = 1 if np_data[row_num][5] == 1.0 else -1
            pred = y_i * vector_prod(weight, np_data[row_num][0:5])
            if pred <= 0:
                weight = weight + (learning_rate * y_i * np_data[row_num][0:5])
    return weight


def predict_test_data(data, weight):
    pred_list = []
    labels = []
    for row in data:
        y_i = 1 if row[5] == 1.0 else -1
        labels.append(y_i)
        pred = vector_prod(weight, row[0:5])
        if pred < 0:
            pred_list.append(-1)
        else:
            pred_list.append(1)
    return pred_list, labels


training_data = read_data('./bank-note/train.csv')
print(training_data)
final_weight = perceptron_algorithm(10, training_data, 0.1)
print("final_weight_vector: ", final_weight)

test_data = read_data('./bank-note/test.csv')
predictions, labels = predict_test_data(test_data, final_weight)
print(predictions)
prediction_Array = np.array(predictions)
print(labels)
labels_array = np.array(labels)
# match = np.count_nonzero(prediction_Array == labels_array)
print("Average Prediction Error: ", np.sum(prediction_Array != labels_array)/len(labels))
count = 0
for i in range(len(labels)):
    if labels[i] != predictions[i]:
        count += 1

print(count/len(labels))
