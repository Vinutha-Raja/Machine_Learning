import math
import random
import pandas as pd
import numpy as np
from Ensemble_Learning.bagged_dt import dt
import matplotlib.pyplot as plt


def draw_samples_without_replacement(m, df):
    num_list = [i for i in range(5000)]
    chosen_list = random.sample(num_list, m)
    new_df = pd.DataFrame(columns=df.columns)
    # print(new_df)
    for r in chosen_list:
        new_df.loc[len(new_df.index)] = df.iloc[r].values
    new_df = new_df.drop(columns=['prediction'], errors='ignore')
    new_df = new_df.drop(columns=['count_1'], errors='ignore')
    new_df = new_df.drop(columns=['count_final'], errors='ignore')

    # print("new_df \n ",new_df)
    return new_df


# def calculate_bias(df):
#     df['bias_1'] = [0] * len(df.index)
#     df['bias_final'] = [0] * len(df.index)
#     col_size = len(df.columns) - 1
#     for i in range(len(df.index)):
#         val = 1 if df.iloc[col_size - 4] == "yes" else -1
#         df.iloc[i, col_size - 1] = math.pow(df.iloc[i, col_size - 3]/100 - val)
#         df.iloc[i, col_size] = math.pow(df.iloc[i, col_size - 2] / 100 - val)
#     return df

def calculate_variance(iter_pred, num):
    # mean_pred = iter_pred.mean(axis=1, dtype=float)
    # iter_pred = iter_pred - iter_pred.mean(axis=1, dtype=float)
    # iter_sum = iter_pred.sum(axis=1, dtype=float)
    # variance = iter_sum/num
    # print(variance)

    # means = [row.mean() for row in iter_pred]
    # squared_errors = [(row - mean) ** 2 for row, mean in zip(iter_pred, means)]
    # variance = [row.mean() for row in squared_errors]
    variance = np.var(iter_pred, axis=1)
    return variance


def calculate_bias(iter_pred, labels, num, final_pred=False):
    # iter_pred = np.array(iter_pred)
    # labels = np.array(labels)
    labels[labels == 'yes'] = 1
    labels[labels == 'no'] = -1
    # mean_pred = iter_pred.mean(axis=1, dtype=float)
    mean_pred = np.array([row.mean() for row in iter_pred])
    # print("mean", mean_pred)
    if final_pred:
        mean_pred = mean_pred - labels
    else:
        mean_pred = mean_pred - labels
    # print("mean1", mean_pred)
    mean_pred = np.square(mean_pred)
    # print("bias", mean_pred)
    return mean_pred


def draw_samples(m, df, drop_cols=False):
    # print(df)
    new_df = pd.DataFrame(columns=df.columns)
    # print(new_df)
    for j in range(m):
        r = random.randint(0, len(df.index) - 1)
        # print(r)
        # print(df.iloc[r].values)
        # new_df.append(df.iloc[[r]], ignore_index=True)
        new_df.loc[len(new_df.index)] = df.iloc[r].values
    if drop_cols:
        new_df = new_df.drop(columns=['prediction'], errors='ignore')
        new_df = new_df.drop(columns=['count_1'], errors='ignore')
        new_df = new_df.drop(columns=['count_final'], errors='ignore')

    # print("new_df \n ",new_df)
    return new_df


def bagging_algorithm(T, data_df, test_data_df, first_iter_pred, last_iter_pred, iter_num):
    # end - set all the attribute details as per bank dataset
    # data_df = dt.read_training_data('/Users/vinutha/Documents/FALL2022/ML/Mach
    # ine_Learning/Ensemble_Learning/bank/train.csv', 'bank', "False")
    # data_df['prediction'] = [0] * len(data_df.index)
    # data_df['count'] = [0] * len(data_df.index)
    # test_data_df = dt.read_training_data(
    #     '/Users/vinutha/Documents/FALL2022/ML/Machine_Learning/Ensemble_Learning/bank/test.csv', 'bank', "False")
    # test_data_df['prediction'] = [0] * len(test_data_df.index)
    # test_data_df['count'] = [0] * len(test_data_df.index)
    for i in range(1, T+1):
        first_iter = False
        last_iter = False
        training_samples = data_df.sample(frac=0.5, replace=True, random_state=i)
        training_samples = training_samples.drop(columns=['prediction'], errors='ignore')
        training_samples = training_samples.drop(columns=['count_1'], errors='ignore')
        training_samples = training_samples.drop(columns=['count_final'], errors='ignore')

        training_error_count = 0
        testing_error_count = 0

        dt.node = dt.constuct_decision_tree(training_samples, 'entropy')
        # predict values for training dataset

        # training_data_size = len(data_df.index)
        # print("data_df before \n", data_df)
        if i == 1:
            first_iter = True
        if i == T:
            last_iter = True

        # predicted_training_df, first_iter_pred, last_iter_pred = dt.predict_labels_by_iter(data_df, dt.node, first_iter_pred, last_iter_pred,
        #                                                   first_iter=first_iter, iter_num=iter_num)

        # print("predicted_training_df \n", predicted_training_df)
        # for ii in range(len(predicted_training_df.index)):
        #     if predicted_training_df.iloc[ii, 16] != predicted_training_df.iloc[ii, 17]:
        #         training_error_count += 1
        # print("test df before \n", test_data_df)

        predicted_test_df, first_iter_pred, last_iter_pred = dt.predict_labels_by_iter(test_data_df, dt.node, first_iter_pred, last_iter_pred,
                                                      first_iter=first_iter, last_iter=last_iter, iter_num=iter_num)

        # print("predicted_test_df\n", predicted_test_df)
        # for j in range(len(predicted_test_df.index)):
        #     if predicted_test_df.iloc[j, 16] != predicted_test_df.iloc[j, 17]:
        #         testing_error_count += 1
        # test_df_size = len(test_data_df.index)

        # print('Iteration ', i, ' Ensemble_Learning/bank/train.csv', "   ", 'Entropy',
        # "       ", dt.max_depth, "     ",
        #       training_error_count / training_data_size, "  ", "||", 'Ensemble_Learning/bank/test.csv', "     ",
        #       testing_error_count / test_df_size)
    #
    # print("final train df: \n", data_df)
    # print("final test_df: \n", test_data_df)
    # train_err_count = (data_df['count_1'] < 0).sum()
    # test_err_count = (test_data_df['count_1'] < 0).sum()
    # print("Training error single tree: ", train_err_count/len(data_df.index))
    # print("Test Data error single tree: ", test_err_count / len(test_data_df.index))
    # train_err_count = (data_df['count_final'] < 0).sum()
    # test_err_count = (test_data_df['count_final'] < 0).sum()
    # print("Training error bagged tree: ", train_err_count / len(data_df.index))
    # print("Test Data error bagged tree: ", test_err_count / len(test_data_df.index))
    # print("first_iter_pred: \n", first_iter_pred)
    # print("last_iter_pred: \n", last_iter_pred)
    return


def repeated_bagging(num_iter):
    dt.max_depth = 16
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

    data_df = dt.read_training_data(
        './bank/train.csv', 'bank', "False")
    # print(data_df)
    # data_df['prediction'] = [0] * len(data_df.index)
    # data_df['count_1'] = [0] * len(data_df.index)
    # data_df['count_final'] = [0] * len(data_df.index)
    test_data_df = dt.read_training_data(
        './bank/test.csv', 'bank', "False")
    # test_data_df['prediction'] = [0] * len(test_data_df.index)
    # test_data_df['count_1'] = [0] * len(test_data_df.index)
    # test_data_df['count_final'] = [0] * len(test_data_df.index)
    first_iter_pred = [[] for _ in range(5000)]
    # print(len(first_iter_pred))
    # print(len(first_iter_pred[0]))
    # last_iter_pred = [[] for _ in range(5000)]
    last_iter_pred = [[0] * num_iter for _ in range(5000)]
    for i in range(num_iter):
        # training_samples = draw_samples_without_replacement(sample_count, data_df)
        training_samples = data_df.sample(frac=0.2, replace=False, random_state=i)
        bagging_algorithm(500, training_samples, test_data_df, first_iter_pred,
                                                            last_iter_pred, i)
    print("first_iter_pred: ", first_iter_pred)
    print("last_iter_pred:", last_iter_pred)
    first_iter_pred = np.array(first_iter_pred)
    last_iter_pred = np.array(last_iter_pred)
    last_iter_pred[last_iter_pred >= 0] = 1
    last_iter_pred[last_iter_pred < 0] = -1

    labels = np.array(test_data_df['label'])
    first_bias = calculate_bias(first_iter_pred, labels, num_iter)
    last_bias = calculate_bias(last_iter_pred, labels, num_iter, final_pred=True)
    first_variance = calculate_variance(first_iter_pred, num_iter)
    last_variance = calculate_variance(last_iter_pred, num_iter)
    print(first_bias, last_bias, first_variance, last_variance)
    print("single tree: bias : {}, variance: {}".format(first_bias.mean(), first_variance.mean()))
    print("bagged tree: bias : {}, variance: {}".format(last_bias.mean(), last_variance.mean()))


repeated_bagging(100)

# single tree: bias : 0.37812999999999203, variance: 0.38031000000000004
# bagged tree: bias : 6036.826008000014, variance: 491.329352

# single tree: bias : 0.3505502222222279, variance: 0.3425964444444444
# bagged tree: bias : 0.37465155555555973, variance: 0.12532177777777778