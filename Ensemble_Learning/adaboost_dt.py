from collections import deque

import pandas as pd
import math
import sys


class TreeNode:
    def __init__(self):
        self.value = None
        self.next_node = None
        self.child_nodes = None


class DecisionTree:
    def __init__(self):
        self.node = None
        self.attribute_index_map = {"buying": 0, "maint": 1, "doors": 2, "persons": 3, "lug_boot": 4, "safety": 5}
        self.attribute_map = {"buying": ["vhigh", "high", "med", "low"],
                         "maint": ["vhigh", "high", "med", "low"],
                         "doors": ["2", "3", "4", "5more"],
                         "persons": ["2", "4", "more"],
                         "lug_boot": ["small", "med", "big"],
                         "safety": ["low", "med", "high"]
                         }

        self.labels_val = ['unacc', 'acc', 'good', 'vgood']
        self.depth = 0
        self.max_depth = 0
        self.attribute_list = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
        self.bank_attribute_list = ["age", "job", "marital", "education", "default", "balance",
                                    "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                                    "previous", "poutcome"]
        self.majority_map = {}

    def get_best_split_attribute(self, df, attribute_list, heuristic_name):
        # print(df)
        attributes = attribute_list
        gains_map = {}
        # print(heuristic_name)
        if heuristic_name == "entropy":
            total_entropy = self.calculate_entropy(df)
            # print(total_entropy)
            for attr in attributes:
                gains_map[attr] = self.calculate_information_gain(df, attr, total_entropy, heuristic_name)
        elif heuristic_name == "majority_error":
            total_majority_error = self.calculate_majority_error(df)
            for attr in attributes:
                gains_map[attr] = self.calculate_information_gain(df, attr, total_majority_error, heuristic_name)
        elif heuristic_name == "gini_index":
            total_gini_index = self.calculate_gini_index(df)
            for attr in attributes:
                gains_map[attr] = self.calculate_information_gain(df, attr, total_gini_index, heuristic_name)
        return max(gains_map, key=gains_map.get)

    def calculate_information_gain(self, df, attribute_name, total_error, heuristic_name):
        if heuristic_name == "entropy":
            expected_error = self.calculate_expected_entropy(df, attribute_name)
        elif heuristic_name == "majority_error":
            expected_error = self.calculate_expected_majority_error(df, attribute_name)
        elif heuristic_name == "gini_index":
            expected_error = self.calculate_expected_gini_index(df, attribute_name)
        information_gain = total_error - expected_error
        return information_gain

    def calculate_expected_entropy(self, df, attribute_name):
        attribute_values = self.attribute_map[attribute_name]
        expected_entropy = 0
        for attr_val in attribute_values:
            attribute_df = df[df[attribute_name] == attr_val]
            entropy = self.calculate_entropy(attribute_df)  # send rows of specific attribute value
            proportion = attribute_df['weight'].sum() / df['weight'].sum()
            expected_entropy += proportion * entropy

        return expected_entropy

    # df is filtered based on a particular value of an attribute
    def calculate_entropy(self, df):
        # get the total rows in df
        # get the count of each label value
        # for each label value calculate the entropy
        # df_size = len(df.index)
        total_weight = df['weight'].sum()
        entropy = 0
        for val in self.labels_val:
            if val in df['label'].unique():
                val_df = df[df['label'] == val]
                val_count = val_df['weight'].sum()
                proportion = val_count / total_weight
                log_val = math.log(proportion, 2) * proportion * (-1)
                entropy += log_val

        return entropy

    def get_max_weighted_value(self, df):
        val_yes = 0
        val_no = 0
        for i in range(len(df)):
            if df.iloc[i, 16] == "yes":
                val_yes += df.iloc[i, 17]
            else:
                val_no += df.iloc[i, 17]
        if val_yes >= val_no:
            return "yes"
        else:
            return "no"

    def id3(self, df, attribute_list, node=None, heuristic_name='entropy', depth=0):
        if not node:
            node = TreeNode()

        # if all the rows have same label. Return a node with that label
        unique_labels = df['label'].unique()
        if len(unique_labels) == 1:
            node.value = unique_labels[0]
            return node

        if not attribute_list:
            node.value = self.get_max_weighted_value(df)
            return node

        self.depth = depth
        # print("Depth: ", self.depth)
        if self.depth == self.max_depth:
            # max depth has been reached so assign the majority value
            node.value = self.get_max_weighted_value(df)
            return node
        # print("attribute_list", attribute_list)
        best_split_attribute = self.get_best_split_attribute(df, attribute_list, heuristic_name)
        # print("best_split_attribute: ", best_split_attribute)
        node.value = best_split_attribute
        node.child_nodes = []

        attribute_values = self.attribute_map.get(best_split_attribute)
        # print(attribute_values)
        for attribute in attribute_values:
            # print(attribute)
            child = TreeNode()
            child.value = attribute
            node.child_nodes.append(child)
            new_df = df[df[best_split_attribute] == attribute]
            if new_df.size == 0:
                child.next_node = self.get_max_weighted_value(df)
            else:
                # if best_split_attribute in attribute_list:
                new_attr_list = attribute_list.copy()
                new_attr_list.remove(best_split_attribute)
                # attribute_list.remove(best_split_attribute)
                child.next_node = self.id3(new_df, new_attr_list, child.next_node, heuristic_name, depth=depth + 1)
        return node

    def constuct_decision_tree(self, df, heuristic):
        # print("here")
        attribute_list = list(self.attribute_map.keys())
        # print(attribute_list)
        self.node = self.id3(df, attribute_list, self.node, heuristic)
        # self.level_order_print_tree(self.node)
        # self.print_decision_tree()

    def convert_numeric_to_binary_attributes(self, df):
        attributes_to_convert = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

        for attr in attributes_to_convert:
            # calculate the median of that attribute
            # update all the attributes to yes if it is more than median else no
            # TODO: check for pdays -1 value
            median = df[attr].median()
            for i in range(len(df)):
                if df.iloc[i, self.attribute_index_map[attr]] > median:
                    df.iloc[i, self.attribute_index_map[attr]] = "yes"
                else:
                    df.iloc[i, self.attribute_index_map[attr]] = "no"
        # print("converted_df")
        # print(df)

    def update_missing_attributes(self, df):
        for attr in self.bank_attribute_list:
            if 'unknown' in df[attr].unique():
                majority_list = df[attr].value_counts().index.tolist()[:2]
                # print(type(majority_list))
                # print(majority_list[0])
                majority = majority_list[0]
                if majority == 'unknown':
                    majority = majority_list[1]
                self.majority_map[attr] = majority
                for i in range(len(df)):
                    if df.iloc[i, self.attribute_index_map[attr]] == 'unknown':
                        df.iloc[i, self.attribute_index_map[attr]] = majority

    def update_test_missing_attributes(self, df):
        # print(self.majority_map)
        for attr in self.bank_attribute_list:
            if 'unknown' in df[attr].unique():
                for i in range(len(df)):
                    if df.iloc[i, self.attribute_index_map[attr]] == 'unknown':
                        df.iloc[i, self.attribute_index_map[attr]] = self.majority_map[attr]

    def read_training_data(self, file_name, data_set, is_unknown="False"):
        df = pd.read_csv(file_name, header=None,
                         names=self.attribute_list)
        # print(df.size)
        if data_set == "bank":
            # Convert numeric attributes to binary
            # [age, balance, day, duration, campaign, pdays, previous]
            self.convert_numeric_to_binary_attributes(df)
            if is_unknown == "True":
                # print(is_unknown)
                if file_name == "bank/train.csv":
                    self.update_missing_attributes(df)
                else:
                    self.update_test_missing_attributes(df)
        # print(df)
        return df

    def read_training_data_without_labels(self, filename):
        df = pd.read_csv(filename, header=None,
                         names=self.attribute_list)
        # print(df)
        return df

    def predict_label_for_row(self, row, node):
        attribute = node.value
        if attribute not in self.attribute_index_map.keys():
            # print(attribute)
            return attribute
        attribute_val = row[self.attribute_index_map[attribute]]
        children = node.child_nodes
        for child in children:
            if child.value == attribute_val:
                if isinstance(child.next_node, str):
                    return child.next_node
                else:
                    # print(child.next_node.value)
                    attribute = self.predict_label_for_row(row, child.next_node)
        return attribute

    # read the training dataset without labels
    # for each row, traverse the tree and find the attribute name and find the corresponding value of the attribute in
    # in the row, and find the child node matching tha value and find the next_node.
    # Find the next attribute until you find the label for it.
    def predict_labels(self, df, train=True):
        for i in range(len(df)):
            col_size = len(df.columns) - 1
            if train:
                last_index = col_size - 2
            else:
                last_index = col_size - 1
            row = [df.iloc[i, j] for j in range(last_index)]
            # print("row size", len(row))
            # print(row)
            # row = [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3], df.iloc[i, 4], df.iloc[i, 5]]
            # print(df.iloc[i, col_size])
            # print(row)
            df.iloc[i, col_size] = self.predict_label_for_row(row, self.node)
            if df.iloc[i, col_size] not in self.labels_val:
                print("after", df.iloc[i, col_size])

        # print("predicted df: ")
        # print(df)
        return df

    # def get_training_data_with_weight(self, training_filename, dataset, isunknown):
    #     data_df = dt.read_training_data(training_filename, dataset, isunknown)
    #     total_samples = len(data_df.index)
    #     weights = [1/total_samples for i in range(total_samples)]
    #     data_df['weight'] = weights
    #     return data_df


if __name__ == "__main__":
    # 1. Data set name
    # 2. Training Filename
    # 3. Test Filename
    # 4. Heuristic(entropy, majority_error, gini_index)
    # 5. Max depth
    # run for max_depth and report the error
    if len(sys.argv) < 7:
        print("Please provide filename and heuristic name to be used ")
        exit(1)
    print(sys.argv)
    dataset = sys.argv[1]
    training_filename = sys.argv[2]
    test_filename = sys.argv[3]
    heuristic_method_name = sys.argv[4]
    max_depth = int(sys.argv[5])
    isunknown = sys.argv[6]
    print("filename           heuristic_name max_depth error_count")
    training_error_count = 0
    testing_error_count = 0

    dt = DecisionTree()
    dt.max_depth = max_depth

    for depth_count in range(max_depth, max_depth+1):
        # read training dataset and construct the decision tree
        data_df = dt.read_training_data(training_filename, dataset, isunknown)
        dt.constuct_decision_tree(data_df, heuristic_method_name)
        # predict values for training dataset
        training_data = data_df
        training_data_size = len(training_data.index)
        # print("training_data_size", training_data_size)
        predicted_training_df = dt.predict_labels(training_data.copy())
        predicted_training_df.drop(columns=['label'])
        predicted_training_df.rename(columns={"prediction": "label"})
        diff_df = training_data.compare(predicted_training_df)

        # predict values for test dataset
        test_data_df = dt.read_training_data(test_filename, dataset, isunknown)
        test_df = test_data_df
        predicted_test_df = dt.predict_labels(test_df.copy())
        diff_test_df = test_df.compare(predicted_test_df)
        test_df_size = len(test_df.index)

        training_error_count += diff_df.shape[0]
        testing_error_count += diff_test_df.shape[0]
        # print("{} {} {} {} )
        print(training_filename, "   ", heuristic_method_name, "       ", dt.max_depth, "     ", diff_df.shape[0]/training_data_size, "  ",      "||", test_filename, "     ", diff_test_df.shape[0]/test_df_size)
        # print(test_filename, "    ", heuristic_method_name, "       ", dt.max_depth, "     ", diff_test_df.shape[0])

    print("Avg prediction error for training dataset : {:.4f}".format(training_error_count/(max_depth * training_data_size)))
    print("Avg prediction error for testing dataset : {:.4f}".format(testing_error_count / (max_depth * test_df_size)))


dt = DecisionTree()


