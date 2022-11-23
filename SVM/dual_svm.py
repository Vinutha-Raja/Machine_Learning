import pandas as pd
import numpy as np
import scipy.optimize as opt


def object_func(x, train_data, train_y):
    res = 0
    res = res - np.sum(x)
    prod = np.multiply(np.reshape(x, (-1, 1)), np.reshape(train_y, (-1, 1)))
    prod1 = np.multiply(prod, train_data)
    res = res + (np.sum(np.matmul(prod1, np.transpose(prod1))) * 0.5)
    return res


def dual_svm(train_data, train_y, c):
    rows = train_data.shape[0]
    bnds = [(0, c)] * rows
    cons = ({'type': 'eq', 'fun': lambda x: np.matmul(np.reshape(x, (1, -1)), np.reshape(train_y, (-1, 1)))[0]})
    x1 = np.zeros(rows)
    res = opt.minimize(lambda x: object_func(x, train_data, train_y), x1,  method='SLSQP', bounds=bnds,
                       constraints=cons,
                       options={'disp': False})
    prod = np.multiply(np.multiply(np.reshape(res.x, (-1, 1)), np.reshape(train_y, (-1, 1))), train_data)
    weight_vec = np.sum(prod, axis=0)
    index = np.where((res.x > 0) & (res.x < c))
    b = np.mean(train_y[index] - np.matmul(train_data[index, :], np.reshape(weight_vec, (-1, 1))))
    weight_vec = weight_vec.tolist()
    weight_vec.append(b)
    weight_vec = np.array(weight_vec)
    return weight_vec


def gaussian_object_func(x, k_xi_xj, train_y):
    res = 0
    res = res - np.sum(x)
    prod = np.multiply(np.reshape(x, (-1, 1)), np.reshape(train_y, (-1, 1)))
    prod1 = np.matmul(prod, np.transpose(prod))
    res = res + (np.sum(np.multiply(prod1, k_xi_xj)) * 0.5)
    return res


def gaussian_kernel_func(x1, x2, gamma):
    x_i = np.tile(x1, (1, x2.shape[0]))
    x_i = np.reshape(x_i, (-1, x1.shape[1]))
    x_j = np.tile(x2, (x1.shape[0], 1))
    k_xi_xj = np.exp(np.sum(np.square(x_i - x_j), axis=1) / -gamma)
    k_xi_xj = np.reshape(k_xi_xj, (x1.shape[0], x2.shape[0]))
    return k_xi_xj


def gaussian_svm(train_data, train_y, C, gamma):
    rows = train_data.shape[0]
    bnds = [(0, C)] * rows
    cons = ({'type': 'eq', 'fun': lambda x: np.matmul(np.reshape(x, (1, -1)), np.reshape(train_y, (-1, 1)))[0]})
    x_1 = np.zeros(rows)
    k_xi_xj = gaussian_kernel_func(train_data, train_data, gamma)
    res = opt.minimize(lambda x: gaussian_object_func(x, k_xi_xj, train_y), x_1, method='SLSQP', bounds=bnds,
                       constraints=cons,
                       options={'disp': False})
    return res.x


def gaussian_prediction_func(gamma, sol, x, x1, y):
    k = gaussian_kernel_func(x, x1, gamma)
    k = np.multiply(np.reshape(y, (-1, 1)), k)
    pred = np.sum(np.multiply(np.reshape(sol, (-1, 1)), k), axis=0)
    pred = np.reshape(pred, (-1, 1))
    return pred


train = pd.read_csv('./bank-note/train.csv', header=None)
# r = train.shape[0]
c = train.shape[1]
train_vals = train.values
train_data = np.copy(train)
train_data[:, c - 1] = 1
train_y = train_vals[:, c - 1]
train_y = 2 * train_y - 1  # converting 0,1 to 1, -1
# print(train_data)
# print(train_data[:, [x for x in range(c - 1)]])
train_data1 = train_data[:, [x for x in range(c - 1)]]
# print(np.reshape/(train_y, (-1, 1)))
test = pd.read_csv('./bank-note/test.csv', header=None)
# r = train.shape[0]
c = test.shape[1]
test_vals = test.values
test_data = np.copy(test)
test_data[:, c - 1] = 1
test_data1 = test_data[:, [x for x in range(c - 1)]]
test_y = test_vals[:, c - 1]
test_y = 2 * test_y - 1  # converting 0,1 to 1, -1

C_vals = [100/873, 500/873, 700/873]
for c in C_vals:
    w = dual_svm(train_data1, train_y, c)
    w1 = np.reshape(w, (5, 1))
    prediction_Array = np.matmul(train_data, w1)
    # print(prediction_Array)
    prediction_Array[prediction_Array > 0] = 1
    prediction_Array[prediction_Array <= 0] = -1
    # print(prediction_Array)
    train_y1 = np.reshape(train_y, (-1, 1))
    train_err_count = np.sum(prediction_Array != train_y1)
    # print(train_err_count)
    test_prediction = np.matmul(test_data, w1)
    test_prediction[test_prediction > 0] = 1
    test_prediction[test_prediction <= 0] = -1
    # print(prediction_Array)
    test_y1 = np.reshape(test_y, (-1, 1))
    test_err_count = np.sum(test_prediction != test_y1)

    print("Dual SVM 3(a)=> C: {}, weight_vector: {}, train_err_count: {}, test_err_count: {}".format(c, w,
                                                                                                       train_err_count / len(
                                                                                                           train_y),
                                                                                                       test_err_count / len(
                                                                                                           test_y)))

# Dual SVM 3(a)=> C: 0.1145475372279496, weight_vector: [-0.94292644 -0.65149178 -0.73372181 -0.04102191  2.49818693],
# train_err_count: 0.026376146788990827, test_err_count: 0.03
# Dual SVM 3(a)=> C: 0.572737686139748, weight_vector: [-1.56393822 -1.01405361 -1.18065221 -0.15651885  3.89920002],
# train_err_count: 0.03096330275229358, test_err_count: 0.034
# Dual SVM 3(a)=> C: 0.8018327605956472, weight_vector: [-2.04254299 -1.28069943 -1.51352039 -0.24906591  4.89412642],
# train_err_count: 0.033256880733944956, test_err_count: 0.036
# Kernel Gaussian
gamma_list = np.array([0.1, 0.5, 1, 5, 100])
sv_map = {}
for c in C_vals:
    for gamma in gamma_list:
        sol = gaussian_svm(train_data1, train_y, c, gamma)
        prediction_Array = gaussian_prediction_func(gamma, sol, train_data1, train_data1, train_y)
        prediction_Array[prediction_Array > 0] = 1
        prediction_Array[prediction_Array <= 0] = -1
        train_y1 = np.reshape(train_y, (-1, 1))
        train_err_count = np.sum(prediction_Array != train_y1)

        sv = np.where(sol > 0)[0]
        test_prediction = gaussian_prediction_func(gamma, sol, train_data1, test_data1, train_y)
        test_prediction[test_prediction > 0] = 1
        test_prediction[test_prediction <= 0] = -1
        test_y1 = np.reshape(test_y, (-1, 1))
        test_err_count = np.sum(test_prediction != test_y1)

        print("Gaussian SVM 3(b)=> C: {}, gamma: {}, train_err_count: {}, test_err_count: {} , support_vector_count: {}".format(c, gamma,
                                                                                                         train_err_count / len(
                                                                                                             train_y),
                                                                                                         test_err_count / len(
                                                                                                             test_y), len(sv)))
        if c == 500/873:
            sv_map[gamma] = sv

sv_map_vals = list(sv_map.values())
for i in range(len(sv_map_vals) - 1):
    overlap_count = len(np.intersect1d(sv_map_vals[i], sv_map_vals[i+1]))
    print("overlap count: {}".format(overlap_count))

#
# Gaussian SVM 3(b)=> C: 0.1145475372279496, gamma: 0.1, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 872
# Gaussian SVM 3(b)=> C: 0.1145475372279496, gamma: 0.5, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 838
# Gaussian SVM 3(b)=> C: 0.1145475372279496, gamma: 1.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 847
# Gaussian SVM 3(b)=> C: 0.1145475372279496, gamma: 5.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 537
# Gaussian SVM 3(b)=> C: 0.1145475372279496, gamma: 100.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 703
# Gaussian SVM 3(b)=> C: 0.572737686139748, gamma: 0.1, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 871
# Gaussian SVM 3(b)=> C: 0.572737686139748, gamma: 0.5, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 844
# Gaussian SVM 3(b)=> C: 0.572737686139748, gamma: 1.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 719
# Gaussian SVM 3(b)=> C: 0.572737686139748, gamma: 5.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 676
# Gaussian SVM 3(b)=> C: 0.572737686139748, gamma: 100.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 588
# Gaussian SVM 3(b)=> C: 0.8018327605956472, gamma: 0.1, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 869
# Gaussian SVM 3(b)=> C: 0.8018327605956472, gamma: 0.5, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 783
# Gaussian SVM 3(b)=> C: 0.8018327605956472, gamma: 1.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 824
# Gaussian SVM 3(b)=> C: 0.8018327605956472, gamma: 5.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 486
# Gaussian SVM 3(b)=> C: 0.8018327605956472, gamma: 100.0, train_err_count: 0.0, test_err_count: 0.002 , support_vector_count: 413
# overlap count: 843
# overlap count: 703
# overlap count: 567
# overlap count: 474
