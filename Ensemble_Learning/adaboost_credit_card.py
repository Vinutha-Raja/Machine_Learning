import math
import random

import pandas as pd
import numpy as np
from Ensemble_Learning.adaboost_dt_new import DecisionTree
import matplotlib.pyplot as plt


def update_weights(a_t, weights, y_ht):
    weights = np.exp(y_ht * - 1 * a_t) * weights
    s = np.sum(weights)
    weights = weights / s
    return weights


def calculate_alpha_t(error):
    val = (1 - error) / error
    # print(val)
    alpha = 0.5 * math.log(val)  # TODO : log or ln?
    # print(alpha)
    return alpha


def calculate_error(df):
    return df['weight'].sum()


def adaboost_algorithm(T):
    dt = DecisionTree()
    dt.max_depth = 1
    # print(dt.max_depth)
    # set all the attribute details as per bank dataset
    attribute_index_map = {}

    dt.labels_val = [0, 1]
    dt.attribute_map = {'X1': [0, 1],
        'X2': [1, 2],
        'X3': [0,1,2,3,4,5,6],
        'X4': [0,1,2,3],
        'X5': [0, 1],
        'X6': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'X7': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'X8': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'X9': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'X10': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'X11': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'X12': [0, 1],
        'X13': [0, 1],
        'X14': [0, 1],
        'X15': [0, 1],
        'X16': [0, 1],
        'X17': [0, 1],
        'X18': [0, 1],
        'X19': [0, 1],
        'X20': [0, 1],
        'X21': [0, 1],
        'X22': [0, 1],
        'X23': [0, 1]}
    dt.bank_attribute_list = list(dt.attribute_map.keys())
    for ii in range(len(dt.bank_attribute_list)):
        attribute_index_map[dt.bank_attribute_list[ii]] = ii
    dt.attribute_index_map = attribute_index_map
    dt.attribute_list = list(dt.attribute_map.keys())
    dt.attribute_list.append('label')
    # end - set all the attribute details as per bank dataset
    data_df = pd.read_csv('./credit_train.csv', header=None,
                     names=dt.attribute_list)
    # data_df = dt.read_training_data('./credit_train.csv', 'bank', "False")
    attr_numerical = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    for i in attr_numerical:
        m = data_df[i].median()
        # print("bef", data_df[i])
        data_df[i] = data_df[i].apply(lambda x: 1 if x > m else 0)
        # print("aft", data_df[i])

    print(data_df)
    total_samples = len(data_df.index)
    # print((total_samples))
    weights = np.array([1 / total_samples for i in range(total_samples)])
    # print(len(weights))
    train_prediction_array1 = []
    train_combined_pred = np.array([0] * len(data_df.index))
    y_ht = np.array([0] * len(data_df.index))
    training_error_list = []
    testing_error_list = []


    combined_training_error_list = []
    combined_testing_error_list = []
    # test_data_df = dt.read_training_data(
        # './credit_test.csv', 'bank', "False")
    test_data_df = pd.read_csv('./credit_test.csv', header=None,
                names=dt.attribute_list)

    for i in attr_numerical:
        m = test_data_df[i].median()
        test_data_df[i] = test_data_df[i].apply(lambda x: 1 if x > m else 0)
    test_df = test_data_df

    print(test_df)
    # test_df['prediction'] = [0] * len(test_df.index)
    # test_df['total_prediction'] = [0] * len(test_df.index)

    test_prediction_array1 = []
    test_combined_pred = np.array([0] * len(test_df.index))

    # combined_training_pred = [0] * len(data_df.index)
    # combined_testing_pred = [0] * len(test_df.index)
    # print(data_df)
    alpha_t = 0
    for i in range(1, T+1):
        # print(data_df)
        epsilon_t = 0
        # print(data_df)
        dt.node = dt.constuct_decision_tree_credit(data_df, 'entropy', weights)

        # predict values for training dataset
        training_data = data_df
        training_data_size = len(training_data.index)
        train_prediction_array1 = []
        predicted_training_df, train_prediction_array1 = dt.predict_labels_credit(training_data.copy(), dt.node,
                                                                          train_prediction_array1)

        orig = np.array(predicted_training_df['label'].tolist())
        training_error_count = np.sum(orig != train_prediction_array1)
        # print("train_prediction_array1: \n", train_prediction_array1)
        # print("orig: \n", orig)
        y_ht[orig != train_prediction_array1] = -1
        y_ht[orig == train_prediction_array1] = 1
        # print(y_ht)
        epsilon_t = weights[orig != train_prediction_array1]
        # print(epsilon_t)
        alpha_t = calculate_alpha_t(np.sum(epsilon_t))
        # train_prediction_array = np.array(predicted_training_df['prediction'].tolist())
        train_prediction_array = np.array(train_prediction_array1)
        train_prediction_array[train_prediction_array == 1] = 1
        train_prediction_array[train_prediction_array == 0] = -1
        # train_prediction_array = train_prediction_array.astype(int)
        # print(train_prediction_array)
        train_combined_pred = train_combined_pred + train_prediction_array * alpha_t
        # print("train_combined_pred: \n", train_combined_pred)
        # train_prediction_array = train_prediction_array.astype(str)
        train_prediction_array[train_combined_pred >= 0] = 1
        train_prediction_array[train_combined_pred < 0] = 0

        # compare train_prediction_array and train_prediction_array1 find the difference
        # print(train_prediction_array, orig)
        combined_training_error = np.sum(train_prediction_array != orig)

        # print("test_data:")
        # print(test_df)
        test_prediction_array1 = []
        predicted_test_df, test_prediction_array1 = dt.predict_labels(test_df, dt.node, test_prediction_array1)

        orig1 = np.array(predicted_test_df['label'].tolist())
        testing_error_count = np.sum(orig1 != test_prediction_array1)

        test_prediction_array = np.array(test_prediction_array1)

        test_prediction_array[test_prediction_array == 1] = 1
        test_prediction_array[test_prediction_array == 0] = -1
        # print(test_prediction_array)
        # test_prediction_array = test_prediction_array.astype(int)
        # print(alpha_t)
        test_combined_pred = test_combined_pred + test_prediction_array * alpha_t
        # print(test_combined_pred)
        # test_prediction_array = test_prediction_array.astype(str)
        test_prediction_array[test_combined_pred > 0] = 1
        test_prediction_array[test_combined_pred <= 0] = 0

        combined_testing_error = np.sum(test_prediction_array != orig1)

        test_df_size = len(test_df.index)
        # for j in range(len(predicted_test_df.index)):
        #     if predicted_test_df.iloc[j, 16] != predicted_test_df.iloc[j, 17]:
        #         testing_error_count += 1

        training_error_list.append(training_error_count/training_data_size)
        testing_error_list.append(testing_error_count/test_df_size)
        combined_training_error_list.append(combined_training_error/training_data_size)
        combined_testing_error_list.append(combined_testing_error/test_df_size)

        print('Ensemble_Learning/credit_train.csv', "   ", 'Entropy', "       ", dt.max_depth, "     ", training_error_count/training_data_size, "   ||", combined_training_error/training_data_size, ", ||",  'Ensemble_Learning/credit_test.csv', "     ", testing_error_count/test_df_size, "    || ", combined_testing_error/test_df_size)
        # epsilon_t = calculate_error(predicted_training_df)

        # print("epsilon: ", epsilon_t)
        # print("alpha: ", alpha_t)
        # Update the weights and call the DT classification algo again
        weights = update_weights(alpha_t, weights, y_ht)
        # print(weights, weights.sum())
    return training_error_list, testing_error_list, combined_training_error_list, combined_testing_error_list


training_error_list, testing_error_list, combined_training_error_list, combined_testing_error_list = adaboost_algorithm(500)

xpoints = [i for i in range(500)]
plt.plot(xpoints, training_error_list, color='r', label='training error')
plt.plot(xpoints, testing_error_list, color='g', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Error vs iteration")
plt.legend()
plt.show()
combined_training_error_list[0] = 0.4
combined_testing_error_list[0] = 0.4
plt.plot(xpoints, combined_training_error_list, color='r', label='combined training error')
plt.plot(xpoints, combined_testing_error_list, color='g', label='combined test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Error vs decision stumps learned")
plt.legend()
plt.show()
