import random
import pandas as pd
from Ensemble_Learning.bagged_dt import dt


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
    for i in range(1, T+1):
        training_samples = draw_samples(m, data_df)
        training_error_count = 0
        testing_error_count = 0

        dt.constuct_random_decision_tree(training_samples, 'entropy', feature_size)
        # predict values for training dataset

        training_data_size = len(data_df.index)
        # print("data_df before \n", data_df)
        predicted_training_df = dt.predict_labels(data_df)
        # print("predicted_training_df \n", predicted_training_df)
        for ii in range(len(predicted_training_df.index)):
            if predicted_training_df.iloc[ii, 16] != predicted_training_df.iloc[ii, 17]:
                training_error_count += 1
        # print("test df before \n", test_data_df)
        predicted_test_df = dt.predict_labels(test_data_df)
        # print("predicted_test_df\n", predicted_test_df)
        for j in range(len(predicted_test_df.index)):
            if predicted_test_df.iloc[j, 16] != predicted_test_df.iloc[j, 17]:
                testing_error_count += 1
        test_df_size = len(test_data_df.index)

        print('Iteration ', i, ' Ensemble_Learning/bank/train.csv', "   ", 'Entropy', "       ", dt.max_depth, "     ",
              training_error_count / training_data_size, "  ", "||", 'Ensemble_Learning/bank/test.csv', "     ",
              testing_error_count / test_df_size)

    print("final train df: \n", data_df)
    print("final test_df: \n", test_data_df)
    train_err_count = (data_df['count'] < 0).sum()
    test_err_count = (test_data_df['count'] < 0).sum()
    print("Training error: ", train_err_count/len(data_df.index))
    print("Test Data error: ", test_err_count / len(test_data_df.index))


for feature_size in [2, 4, 6]:
    print("Feature size : ", feature_size)
    bagging_algorithm(500, 2500, feature_size)


