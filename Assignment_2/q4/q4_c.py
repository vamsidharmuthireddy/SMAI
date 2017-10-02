from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model

import argparse, os, sys

def get_input_data(filename):

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(int(line[-1]))
            X.append([float(x) for x in line[:-1]])
    X = np.asarray(X); Y = np.asarray(Y)

    return X, Y

def get_test_data(filename, input_dim):

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            if len(line) == input_dim:
                X.append([float(x) for x in line])
                Y = []
            else:
                Y.append(int(line[-1]))
                X.append([float(x) for x in line[:-1]])

    X = np.asarray(X); Y = np.asarray(Y)

    return X, Y

def calculate_accuracy(X_test, Y_test, elastic_regr_model):
    prediction = elastic_regr_model.predict(X_test)
    prediction = list(map(lambda x: 1 if x > 0.5 else 0, prediction))
    # print sum(list([1 for x in zip(prediction, Y_test) if x[0] == x[1]]))/float(len(prediction))
    accuracy = 0
    if Y_test.shape[0] != 0:
        accuracy = sum((list(map(lambda x, y: 1 if x == y else 0, prediction, Y_test))))/float(len(prediction))

    return accuracy, prediction

def elastic_regr(X_data, Y_data, alpha):
    reg = linear_model.ElasticNet(alpha=alpha)
    reg.fit(X_train, Y_train)
    return reg


if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]

    X, Y = get_input_data(train)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=42)

    X_test, Y_test = get_test_data(test, X_train.shape[1])

    print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape

    elastic_regr_model = []
    alpha_count = 1000.0
    elastic_regr_model = [elastic_regr(X_train, Y_train, alpha/alpha_count) for alpha in range(1, int(alpha_count))]
    out = [calculate_accuracy(X_val, Y_val, regr_model) for regr_model in elastic_regr_model]

    accuracy, prediction = zip(*out)

    max_accuracy = max(accuracy)
    max_accuracy_indices = [i for i, x in enumerate(accuracy) if x == max_accuracy]
    max_accuracy_index = max_accuracy_indices[-1]
    best_alpha = (max_accuracy_index + 1) / alpha_count

    print max_accuracy, best_alpha

    elastic_regr_model = elastic_regr(X_train, Y_train, best_alpha)
    final_accuracy, final_predictions = calculate_accuracy(X_test, Y_test, elastic_regr_model)

    print final_accuracy
    # for i in final_predictions:
    #     print i
