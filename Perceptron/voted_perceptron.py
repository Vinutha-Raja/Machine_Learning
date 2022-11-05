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


def voted_perceptron_algorithm(epoch, data, learning_rate):
    weight_vector_list = []
    weight = [0, 0, 0, 0, 0]
    c = 0
    count = 0
    for i in range(epoch):
        np_data = np.array(data)
        # np.random.shuffle(np_data)
        for row_num in range(len(np_data)):
            y_i = 1 if np_data[row_num][5] == 1.0 else -1
            pred = y_i * vector_prod(np.transpose(weight), np_data[row_num][0:5])
            if pred <= 0:
                weight_vector_list.append((weight, c))
                # c_list.append(c)
                weight = weight + (learning_rate * y_i * np_data[row_num][0:5])
                c = 1
            else:
                count += 1
                c += 1

    return weight, weight_vector_list, count


def predict_test_data(data, w_c_list):
    pred_list = []
    label = []
    for row in data:
        y_i = 1 if row[5] == 1 else -1
        label.append(y_i)
        c_sum = 0
        for w_c in w_c_list:
            vec_prod = vector_prod(np.transpose(w_c[0]), row[0:5])
            if vec_prod <= 0:
                first_sign = -1
            else:
                first_sign = 1
            c_sum += first_sign * w_c[1]
        if c_sum < 0:
            pred_list.append(-1)
        else:
            pred_list.append(1)

    return pred_list, label


training_data = read_data('./bank-note/train.csv')
print(training_data)
final_weight, weight_c_list, correct_pred = voted_perceptron_algorithm(10, training_data, 0.1)
print("final_weight_vector:", final_weight)
print("weight c list: ", weight_c_list)
print("Number of correct predictions: ", correct_pred)

test_data = read_data('./bank-note/test.csv')
predictions, labels = predict_test_data(test_data, weight_c_list)
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
