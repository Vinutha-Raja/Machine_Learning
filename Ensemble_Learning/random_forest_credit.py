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
    # for ii in range(len(dt.bank_attribute_list)):
    #     attribute_index_map[dt.bank_attribute_list[ii]] = ii
    # dt.attribute_index_map = attribute_index_map
    dt.labels_val = [0, 1]
    dt.attribute_map = {'X1': [0, 1],
                        'X2': [1, 2],
                        'X3': [0, 1, 2, 3, 4, 5, 6],
                        'X4': [0, 1, 2, 3],
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
    attr_numerical = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    for i in attr_numerical:
        med = data_df[i].median()
        # print("bef", data_df[i])
        data_df[i] = data_df[i].apply(lambda x: 1 if x > med else 0)

    data_df['prediction'] = [0] * len(data_df.index)
    data_df['count'] = [0] * len(data_df.index)
    test_data_df = pd.read_csv('./credit_test.csv', header=None,
                               names=dt.attribute_list)
    for i in attr_numerical:
        med = test_data_df[i].median()
        test_data_df[i] = test_data_df[i].apply(lambda x: 1 if x > med else 0)

    # test_data_df = dt.read_training_data(
    #     '/Users/vinutha/Documents/FALL2022/ML/Machine_Learning/Ensemble_Learning/bank/test.csv', 'bank', "False")
    test_data_df['prediction'] = [0] * len(test_data_df.index)
    test_data_df['count'] = [0] * len(test_data_df.index)

    train_err_list = []
    test_error_list = []
    # train_prediction_array = np.array([0] * 5000)
    train_combined_pred = np.array([0] * len(data_df.index))
    # test_prediction_array = np.array([0] * 5000)
    test_combined_pred = np.array([0] * len(test_data_df.index))

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
        train_prediction_array[train_prediction_array == 1] = 1
        train_prediction_array[train_prediction_array == 0] = -1
        train_prediction_array = train_prediction_array.astype(int)
        train_combined_pred = train_combined_pred + train_prediction_array
        train_prediction_array = train_prediction_array.astype(str)
        train_prediction_array[train_combined_pred > 0] = 1
        train_prediction_array[train_combined_pred <= 0] = 0
        # predicted_training_df['prediction'] = train_prediction_array
        for j in range(len(predicted_training_df.index)):
            if predicted_training_df.iloc[j, 24] != predicted_training_df.iloc[j, 25]:
                training_error_count += 1

        # predicted_training_df['prediction'] = pd.Series(train_prediction_array)
        # training_error_count = predicted_training_df.apply(lambda row: 1 if row['label'] != row['prediction'] else 0,
        #                                                    axis=1)
        # training_error_count = training_error_count.sum()



        # print("test df before \n", test_data_df)
        predicted_test_df = dt.predict_labels(test_data_df, dt.node)

        test_prediction_array = np.array(predicted_test_df['prediction'].tolist())
        test_prediction_array[test_prediction_array == 1] = 1
        test_prediction_array[test_prediction_array == 0] = -1
        test_prediction_array = test_prediction_array.astype(int)
        test_combined_pred = test_combined_pred + test_prediction_array
        test_prediction_array = test_prediction_array.astype(str)
        test_prediction_array[test_combined_pred > 0] = 1
        test_prediction_array[test_combined_pred <= 0] = -1
        # predicted_test_df['prediction'] = test_prediction_array
        for j in range(len(predicted_test_df.index)):
            if predicted_test_df.iloc[j, 24] != predicted_test_df.iloc[j, 25]:
                testing_error_count += 1

        # predicted_test_df['prediction'] = pd.Series(test_prediction_array)
        # testing_error_count = predicted_test_df.apply(lambda row: 1 if row['label'] != row['prediction'] else 0,
        #                                                    axis=1)
        # testing_error_count = testing_error_count.sum()

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


xpoints = [i for i in range(50)]

# for feature_size in [2, 4, 6]:
#     print("Feature size : ", feature_size)
train_err_count1, test_err_count1 = bagging_algorithm(50, 2500, 2)
train_err_count2, test_err_count2 = bagging_algorithm(50, 2500, 4)
train_err_count3, test_err_count3 = bagging_algorithm(50, 2500, 6)

fig = plt.figure()
fig.suptitle('Credit Random Forest - feature size 2')
plt.plot(xpoints, train_err_count1, color='g', label='training error')
plt.plot(xpoints, test_err_count1, color='r', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Random Forest - feature size 2")
plt.legend()
plt.show()
fig.savefig('rf_credit_2.png')

fig = plt.figure()
fig.suptitle('Credit Random Forest - feature size 4')
plt.plot(xpoints, train_err_count2, color='g', label='training error')
plt.plot(xpoints, test_err_count2, color='r', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Random Forest - feature size 4")
plt.legend()
plt.show()
fig.savefig('rf_credit_4.png')


fig = plt.figure()
fig.suptitle('Credit Random Forest - feature size 6')
plt.plot(xpoints, train_err_count3, color='g', label='training error')
plt.plot(xpoints, test_err_count3, color='r', label='test error')
plt.xlabel("Iteration")
plt.ylabel("Prediction error")
plt.title("Random Forest - feature size 6")
plt.legend()
plt.show()
fig.savefig('rf_credit_6.png')





