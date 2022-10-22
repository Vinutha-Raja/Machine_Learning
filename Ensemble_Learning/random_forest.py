import random
import pandas as pd
from Ensemble_Learning.bagged_dt import dt
import matplotlib.pyplot as plt
import numpy as np


def draw_samples(m, df):
    # print(df)
    new_df = pd.DataFrame(columns=df.columns)
    # print(new_df)
    for j in range(m):
        r = random.randint(0, len(df.index) - 1)
        # print(r)
        # print(df.iloc[r].values)
        # new_df.append(df.iloc[[r]], ignore_index=True)
        new_df.loc[len(new_df.index)] = df.iloc[r].values
    new_df = new_df.drop(columns=['prediction'])
    new_df = new_df.drop(columns=['count'])
    # print("new_df \n ",new_df)
    return new_df


def bagging_algorithm(T, m, feature_size):
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
    # end - set all the attribute details as per bank dataset
    data_df = dt.read_training_data('/Users/vinutha/Documents/FALL2022/ML/Machine_Learning/Ensemble_Learning/bank/train.csv', 'bank', "False")
    data_df['prediction'] = [0] * len(data_df.index)
    data_df['count'] = [0] * len(data_df.index)
    test_data_df = dt.read_training_data(
        '/Users/vinutha/Documents/FALL2022/ML/Machine_Learning/Ensemble_Learning/bank/test.csv', 'bank', "False")
    test_data_df['prediction'] = [0] * len(test_data_df.index)
    test_data_df['count'] = [0] * len(test_data_df.index)

    train_err_list = []
    test_error_list = []
    # train_prediction_array = np.array([0] * 5000)
    train_combined_pred = np.array([0] * 5000)
    # test_prediction_array = np.array([0] * 5000)
    test_combined_pred = np.array([0] * 5000)

    for i in range(1, T+1):
        # training_samples = draw_samples(m, data_df)
        training_samples = data_df.sample(frac=0.5, replace=True, random_state=i)
        training_samples = training_samples.drop(columns=['prediction'])
        training_samples = training_samples.drop(columns=['count'])
        training_error_count = 0
        testing_error_count = 0

        dt.node = dt.constuct_random_decision_tree(training_samples, 'entropy', feature_size)
        # predict values for training dataset

        training_data_size = len(data_df.index)
        # print("data_df before \n", data_df)
        predicted_training_df = dt.predict_labels(data_df, dt.node)
        # print("predicted_training_df \n", predicted_training_df)

        train_prediction_array = np.array(predicted_training_df['prediction'].tolist())
        train_prediction_array[train_prediction_array == 'yes'] = 1
        train_prediction_array[train_prediction_array == 'no'] = -1
        train_prediction_array = train_prediction_array.astype(int)
        train_combined_pred = train_combined_pred + train_prediction_array
        train_prediction_array = train_prediction_array.astype(str)
        train_prediction_array[train_combined_pred > 0] = 'yes'
        train_prediction_array[train_combined_pred <= 0] = 'no'
        predicted_training_df['prediction'] = pd.Series(train_prediction_array)
        training_error_count = predicted_training_df.apply(lambda row: 1 if row['label'] != row['prediction'] else 0, axis=1)
        training_error_count = training_error_count.sum()



        # print("test df before \n", test_data_df)
        predicted_test_df = dt.predict_labels(test_data_df, dt.node)

        test_prediction_array = np.array(predicted_test_df['prediction'].tolist())
        test_prediction_array[test_prediction_array == 'yes'] = 1
        test_prediction_array[test_prediction_array == 'no'] = -1
        test_prediction_array = test_prediction_array.astype(int)
        test_combined_pred = test_combined_pred + test_prediction_array
        test_prediction_array = test_prediction_array.astype(str)
        test_prediction_array[test_combined_pred > 0] = 'yes'
        test_prediction_array[test_combined_pred <= 0] = 'no'
        predicted_test_df['prediction'] = pd.Series(test_prediction_array)
        testing_error_count = predicted_test_df.apply(lambda row: 1 if row['label'] != row['prediction'] else 0,
                                                           axis=1)
        testing_error_count = testing_error_count.sum()

        # print("predicted_test_df\n", predicted_test_df)
        # for j in range(len(predicted_test_df.index)):
        #     if predicted_test_df.iloc[j, 16] != predicted_test_df.iloc[j, 17]:
        #         testing_error_count += 1

        test_df_size = len(test_data_df.index)

        print('Iteration ', i, ' Ensemble_Learning/bank/train.csv', "   ", 'Entropy', "       ", dt.max_depth, "     ",
              training_error_count / training_data_size, "  ", "||", 'Ensemble_Learning/bank/test.csv', "     ",
              testing_error_count / test_df_size)
        train_err_list.append(training_error_count / training_data_size)
        test_error_list.append(testing_error_count / test_df_size)

    print("final train df: \n", data_df)
    print("final test_df: \n", test_data_df)
    train_err_count = (data_df['count'] < 0).sum()
    test_err_count = (test_data_df['count'] < 0).sum()
    print("Training error: ", train_err_count/len(data_df.index))
    print("Test Data error: ", test_err_count / len(test_data_df.index))
    return train_err_list, test_error_list


xpoints = [i for i in range(500)]

# for feature_size in [2, 4, 6]:
#     print("Feature size : ", feature_size)
train_err_count1, test_err_count1 = bagging_algorithm(500, 2500, 2)
train_err_count2, test_err_count2 = bagging_algorithm(500, 2500, 4)
train_err_count3, test_err_count3 = bagging_algorithm(500, 2500, 6)

plt.plot(xpoints, train_err_count1, color='g', label='training error')
plt.plot(xpoints, test_err_count1, color='r', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Random Forest - feature size 2")
plt.legend()
plt.show()

plt.plot(xpoints, train_err_count2, color='g', label='training error')
plt.plot(xpoints, test_err_count2, color='r', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Random Forest - feature size 4")
plt.legend()
plt.show()

plt.plot(xpoints, train_err_count3, color='g', label='training error')
plt.plot(xpoints, test_err_count3, color='r', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Random Forest - feature size 6")
plt.legend()
plt.show()




