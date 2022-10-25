import pandas as pd
import numpy as np

dataset = pd.read_csv('./cc_data.csv', header=None)
dataset.drop(columns=dataset.columns[0], axis=1, inplace=True)

sample = dataset.sample(frac=1, replace=False)
training_data = sample.iloc[:24000]
testing_data = sample.iloc[24000:]
training_data.to_csv('./credit_train.csv', header=None, index=False)
testing_data.to_csv('./credit_test.csv', header=None, index=False)