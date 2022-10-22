import math
import random

import pandas as pd
import numpy as np
from Ensemble_Learning.adaboost_dt import DecisionTree
import matplotlib.pyplot as plt


def update_weights(df, a_t):
    weight_sum = 0
    for i in range(len(df)):
        y_ht = 1 if df.iloc[i, 16] == df.iloc[i, 18] else -1
        if y_ht == -1:
            exp_val = math.exp(a_t * -1 * y_ht)
            df.iloc[i, 17] = df.iloc[i, 17] * exp_val
        weight_sum += df.iloc[i, 17]

    for i in range(len(df)):
        df.iloc[i, 17] = df.iloc[i, 17] / weight_sum
    return df


def calculate_alpha_t(error):
    val = (1 - error) / error
    # print(val)
    alpha = 0.5 * math.log(val)  # TODO : log or ln?
    return alpha


def calculate_error(df):
    return df['weight'].sum()


def adaboost_algorithm(T):
    dt = DecisionTree()
    dt.max_depth = 1
    # print(dt.max_depth)
    # set all the attribute details as per bank dataset
    attribute_index_map = {}
    for ii in range(len(dt.bank_attribute_list)):
        attribute_index_map[dt.bank_attribute_list[ii]] = ii
    dt.attribute_index_map = attribute_index_map
    dt.labels_val = ["yes", "no"]
    dt.attribute_map = {'age': ['yes', 'no'],
                        'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
                                "student",
                                "blue-collar", "self-employed", "retired", "technician", "services"],
                        'marital': ["married", "divorced", "single"],
                        'education': ["unknown", "secondary", "primary", "tertiary"], 'default': ["yes", "no"],
                        'balance': ["yes", "no"], 'housing': ["yes", "no"], 'loan': ["yes", "no"],
                        'contact': ["unknown", "telephone", "cellular"], 'day': ["yes", "no"],
                        'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                  "dec"], 'duration': ["yes", "no"], 'campaign': ["yes", "no"],
                        'pdays': ["yes", "no"], 'previous': ["yes", "no"],
                        'poutcome': ["unknown", "other", "failure", "success"]}
    dt.attribute_list = ["age", "job", "marital", "education", "default", "balance",
                         "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                         "previous", "poutcome", "label"]
    # end - set all the attribute details as per bank dataset

    data_df = dt.read_training_data('/Users/vinutha/Documents/FALL2022/ML/Machine_Learning/Ensemble_Learning/bank/train.csv', 'bank', "False")
    # print(data_df)
    total_samples = len(data_df.index)
    weights = [1 / total_samples for i in range(total_samples)]
    data_df['weight'] = weights
    data_df['prediction'] = [0] * len(data_df.index)
    data_df['total_prediction'] = [0] * len(data_df.index)
    training_error_list = []
    testing_error_list = []


    combined_training_error_list = []
    combined_testing_error_list = []
    test_data_df = dt.read_training_data(
        '/Users/vinutha/Documents/FALL2022/ML/Machine_Learning/Ensemble_Learning/bank/test.csv', 'bank', "False")
    test_df = test_data_df
    test_df['prediction'] = [0] * len(test_df.index)
    test_df['total_prediction'] = [0] * len(test_df.index)
    combined_training_pred = [0] * len(data_df.index)
    combined_testing_pred = [0] * len(test_df.index)
    # print(data_df)
    alpha_t = 0
    for i in range(1, T+1):
        # print(data_df)
        training_error_count = 0
        testing_error_count = 0
        combined_training_error = 0
        combined_testing_error = 0

        dt.node = dt.constuct_decision_tree(data_df, 'entropy')
        # predict values for training dataset
        training_data = data_df
        training_data_size = len(training_data.index)
        # print("training_Df :")
        # print(training_data)
        predicted_training_df = dt.predict_labels(training_data.copy(), dt.node)
        # print("datadf: \n", data_df)
        # print("predicted: \n", predicted_training_df)
        # print(len(predicted_training_df.index))
        epsilon_t = 0
        for ii in range(len(predicted_training_df.index)):
            if predicted_training_df.iloc[ii, 16] != predicted_training_df.iloc[ii, 18]:
                training_error_count += 1
                epsilon_t += predicted_training_df.iloc[ii, 17]

        pred = np.array(predicted_training_df['prediction'].tolist())
        pred[pred == 'yes'] = 1
        pred[pred == 'no'] = -1
        pred = pred.astype(int)
        combined_training_pred = combined_training_pred + pred * alpha_t
        pred = pred.astype(str)
        pred[combined_training_pred > 0] = 'yes'
        pred[combined_training_pred <= 0] = 'no'
        predicted_training_df['prediction'] = pd.Series(pred)
        no_error_count = predicted_training_df.apply(lambda row: 1 if row['label'] == row['prediction'] else 0, axis=1)
        combined_training_error = 1 - no_error_count.sum() / training_data_size



            # if predicted_training_df.iloc[ii, 19] > 0:
            #     val = "yes"
            # elif predicted_training_df.iloc[ii, 19] <= 0:
            #     val = "no"
            # else:
            #     val = random.choice(["yes", "no"])
            #
            # if val != predicted_training_df.iloc[ii, 16]:
            #     combined_training_error += 1



        # print("predicted_df: \n", predicted_training_df)
        # predict values for test dataset



        # print("test_data:")
        # print(test_df)
        predicted_test_df = dt.predict_labels(test_df, dt.node, False)
        test_df_size = len(test_df.index)
        for j in range(len(predicted_test_df.index)):
            if predicted_test_df.iloc[j, 16] != predicted_test_df.iloc[j, 17]:
                testing_error_count += 1
        test_pred = np.array(predicted_test_df['prediction'].tolist())
        test_pred[test_pred == 'yes'] = 1
        test_pred[test_pred == 'no'] = -1
        test_pred = test_pred.astype(int)
        combined_testing_pred = combined_testing_pred + test_pred * alpha_t
        test_pred = test_pred.astype(str)
        test_pred[combined_testing_pred > 0] = 'yes'
        test_pred[combined_testing_pred <= 0] = 'no'
        predicted_test_df['prediction'] = pd.Series(pred)
        no_error_count = predicted_test_df.apply(lambda row: 1 if row['label'] == row['prediction'] else 0, axis=1)
        combined_testing_error = 1 - no_error_count.sum() / test_df_size

            # if predicted_test_df.iloc[j, 18] > 0:
            #     val = "yes"
            # elif predicted_test_df.iloc[j, 18] < 0:
            #     val = "no"
            # else:
            #     val = random.choice(["yes", "no"])
            #
            # if val != predicted_test_df.iloc[j, 16]:
            #     combined_testing_error += 1

        training_error_list.append(training_error_count/training_data_size)
        testing_error_list.append(testing_error_count/test_df_size)
        combined_training_error_list.append(combined_training_error)
        combined_testing_error_list.append(combined_testing_error)

        print('Ensemble_Learning/bank/train.csv', "   ", 'Entropy', "       ", dt.max_depth, "     ", training_error_count/training_data_size, "   ||", combined_training_error, ", ||",  'Ensemble_Learning/bank/test.csv', "     ", testing_error_count/test_df_size, "    || ", combined_testing_error)
        # epsilon_t = calculate_error(predicted_training_df)
        alpha_t = calculate_alpha_t(epsilon_t)
        # print("epsilon: ", epsilon_t)
        # print("alpha: ", alpha_t)
        # Update the weights and call the DT classification algo again
        data_df = update_weights(predicted_training_df, alpha_t)
    return training_error_list, testing_error_list, combined_training_error_list, combined_testing_error_list


training_error_list, testing_error_list, combined_training_error_list, combined_testing_error_list = adaboost_algorithm(50)

xpoints = [i for i in range(50)]
plt.plot(xpoints, training_error_list, color='r', label='training error')
plt.plot(xpoints, testing_error_list, color='g', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.legend()
plt.show()

plt.plot(xpoints, combined_training_error_list, color='r', label='combined training error')
plt.plot(xpoints, combined_testing_error_list, color='g', label='combined test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.legend()
plt.show()
