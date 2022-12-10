import pandas as pd
import numpy as np


def generate_weights(s, width, is_random=True):
    W1 = np.random.normal(size=(s, width)) if is_random else np.zeros((s, width))
    b1 = np.random.normal(size=width) if is_random else np.zeros(width)

    W2 = np.random.normal(size=(width, width)) if is_random else np.zeros((width, width))
    b2 = np.random.normal(size=width) if is_random else np.zeros(width)

    W3 = np.random.normal(size=(width, 1)) if is_random else np.zeros((width, 1))
    b3 = np.random.normal(size=1) if is_random else np.zeros(1)

    return W1, W2, W3, b1, b2, b3


def sigmoid_func(x):
    den = (1 + np.exp(-x))
    return 1 / den


def calculate_mean_error(scores, y):
    return np.mean(0.5 * ((scores - y) ** 2))


def forward_func(X, W1, W2, W3, b1, b2, b3):
    S1 = np.dot(X, W1) + b1
    Z1 = sigmoid_func(S1)
    S2 = np.dot(Z1, W2) + b2
    Z2 = sigmoid_func(S2)
    scores = np.dot(Z2, W3) + b3
    cache = (S1, Z1, S2, Z2)
    return scores, cache, W1, W2, W3, b1, b2, b3


def backward_func(X, y, score, c, W1, W2, W3):
    S1, Z1, S2, Z2 = c
    d_y = (score - y).reshape((1, 1))
    d_W3 = np.dot(Z2.reshape((1, -1)).T, d_y)
    d_b3 = np.sum(d_y, axis=0)
    d_Z2 = np.dot(d_y, W3.reshape((-1, 1)).T)
    dsig_2 = sigmoid_func(S2) * (1 - sigmoid_func(S2)) * d_Z2
    d_W2 = np.dot(Z1.reshape((1, -1)).T, dsig_2)
    d_b2 = np.sum(dsig_2, axis=0)
    d_Z1 = np.dot(dsig_2, W2.T)
    dsig_1 = sigmoid_func(S1) * (1 - sigmoid_func(S1)) * d_Z1
    d_W1 = np.dot(X.reshape((1, -1)).T, dsig_1)
    d_b1 = np.sum(dsig_1, axis=0)
    return d_W1, d_W2, d_W3, d_b1, d_b2, d_b3


def train_func(X_data, Y_data, width, lr=0.1, is_random=True, lr_sched=None, T=100):
    r, c = X_data.shape
    id = np.arange(r)
    prev = 0
    error_list = []
    W1, W2, W3, b1, b2, b3 = generate_weights(c, width, is_random=is_random)

    for t in range(T):
        np.random.shuffle(id)
        if lr_sched is None:
            r_t = lr
        else:
            r_t = lr_sched[t]

        for i in id:
            X = X_data[i, :].reshape((1, -1))
            score, ca, W1, W2, W3, b1, b2, b3 = forward_func(X, W1, W2, W3, b1, b2, b3)
            d_W1, d_W2, d_W3, d_b1, d_b2, d_b3 = backward_func(X, Y_data[i], score, ca, W1, W2, W3)
            W1 -= r_t * d_W1
            b1 -= r_t * d_b1
            W2 -= r_t * d_W2
            b2 -= r_t * d_b2
            W3 -= r_t * d_W3
            b3 -= r_t * d_b3
        scores, _, W1, W2, W3, b1, b2, b3 = forward_func(X_data, W1, W2, W3, b1, b2, b3)
        new = calculate_mean_error(scores, Y_data)
        diff = abs(prev - new)
        prev = new
        error_list.append(new)
        # threshold check
        if diff < 1e-6:
            break
    return error_list, W1, W2, W3, b1, b2, b3


def fit(X, W1, W2, W3, b1, b2, b3):
    score, _, W1, W2, W3, b1, b2, b3 = forward_func(X, W1, W2, W3, b1, b2, b3)
    s = np.sign(score.flatten())
    return s


def convert_labels(y):
    labels = y.copy()
    labels[labels == 0] = -1
    return labels


training_data = pd.read_csv('./bank-note 2/train.csv', header=None)
testing_data = pd.read_csv('./bank-note 2/test.csv', header=None)

train_x = training_data.iloc[:, 0:4].to_numpy()
train_y = convert_labels(training_data.iloc[:, 4].to_numpy())

test_x = testing_data.iloc[:, 0:4].to_numpy()
test_y = convert_labels(testing_data.iloc[:, 4].to_numpy())

# 2A

X = np.array([[1, 1]])
Y = np.array([1])
W1 = np.array([[-2, 2], [-3, 3]])
b1 = np.array([-1, 1])
W2 = np.array([[-2, 2], [-3, 3]])
b2 = np.array([-1, 1])
W3 = np.array([[2], [-1.5]])
b3 = np.array([-1])

s, c, W1, W2, W3, b1, b2, b3 = forward_func(X, W1, W2, W3, b1, b2, b3)
d_W1, d_W2, d_W3, d_b1, d_b2, d_b3 = backward_func(X, Y, s, c, W1, W2, W3)

# Compare with threory results
print("FORWARD PASS: Score: {}, S1: {}, Z1: {}, S2: {}, Z2: {}".format(s, c[0], c[1], c[2], c[3]))
print("BACKWARD PASS: d_W1 = {}, d_b1 = {}, d_W2 = {}, d_b2 = {}, d_W3 = {}, d_b3 = {}".format(d_W1, d_b1, d_W2, d_b2, d_W3,
                                                                                         d_b3))


# 2B
d = 0.1
T = 100
t = np.arange(T)
r_0 = 0.1
lr_sched = r_0 / (1 + (r_0/d)*t)
widths = np.array([5, 10, 25, 50, 100])

for w in widths:
    e, W1, W2, W3, b1, b2, b3 = train_func(train_x, train_y, w, is_random=True, lr_sched=lr_sched)
    train_prediction = fit(train_x, W1, W2, W3, b1, b2, b3)
    train_err = np.sum(train_prediction != train_y)
    train_err /= train_y.shape[0]

    y_test_prediction = fit(test_x, W1, W2, W3, b1, b2, b3)
    test_err = np.sum(y_test_prediction != test_y)
    test_err /= test_y.shape[0]
    print("Width: {}, Training error: {}, Testing error: {}".format(w, train_err, test_err))

# 2C
d = 0.1
T = 100
r_0 = 0.1
t = np.arange(T)+1
r_ = r_0 / (1 + (r_0 / d) * t)
widths = np.array([5, 10, 25, 50, 100])

for w in widths:
    e, W1, W2, W3, b1, b2, b3 = train_func(train_x, train_y, w, is_random=False, lr_sched=lr_sched)
    train_prediction = fit(train_x, W1, W2, W3, b1, b2, b3)
    train_err = np.sum(train_prediction != train_y)
    train_err /= train_y.shape[0]

    y_test_prediction = fit(test_x, W1, W2, W3, b1, b2, b3)
    test_err = np.sum(y_test_prediction != test_y)
    test_err /= test_y.shape[0]
    print("Width: {}, Training error: {}, Testing error: {}".format(w, train_err, test_err))

