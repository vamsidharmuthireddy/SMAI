"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM using scikit-learn.
Skeleton code for implementing SVM classifier for a
character recognition dataset having precomputed features for
each character.

Dataset is taken from: https://archive.ics.uci.edu/ml/datasets/letter+recognition

Remember
--------
1) SVM algorithms are not scale invariant.
"""

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

import argparse, os, sys

def get_input_data(filename):
    """
    Function to read the input data from the letter recognition data file.

    Parameters
    ----------
    filename: The path to input data file

    Returns
    -------
    X: The input for the SVM classifier of the shape [n_samples, n_features].
       n_samples is the number of data points (or samples) that are to be loaded.
       n_features is the length of feature vector for each data point (or sample).
    Y: The labels for each of the input data point (or sample). Shape is [n_samples,].

    """

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    """
    An important part is missing here. Corresponding to point (1) in "Remember".
    ===========================================================================
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    """
    ===========================================================================
    """

    return X, Y

def calculate_metrics(predictions, labels):
    """
    Function to calculate the precision, recall and F-1 score.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    precision: true_positives / (true_positives + false_positives)
    recall: true_positives / (true_positives + false_negatives)
    f1: 2 * (precision * recall) / (precision + recall)
    ===========================================================================
    """
    precision = precision_score

    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    """
    ===========================================================================
    """

    return precision, recall, f1

def calculate_accuracy(predictions, labels):
    """
    Function to calculate the accuracy for a given set of predictions and
    corresponding labels.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    accuracy: Fraction of total samples that have correct predictions (same as
    true label)

    """
    return accuracy_score(labels, predictions)

def SVM(train_data, train_labels, test_data, test_labels, kernel='linear'):
    """
    Function to create, train and test the one-vs-all SVM using scikit-learn.

    Parameters
    ----------
    train_data: Numpy ndarray of shape [n_train_samples, n_features]
    train_labels: Numpy ndarray of shape [n_train_samples,]
    test_data: Numpy ndarray of shape [n_test_samples, n_features]
    test_labels: Numpy ndarray of shape [n_test_samples,]
    kernel: linear (default)
            Which kernel to use for the SVM

    Returns
    -------
    accuracy: Accuracy of the model on the test data
    top_predictions: Top predictions for each test sample
    precision: The precision score for the test data
    recall: The recall score for the test data
    f1: The F1-score for the test data

    """

    clf = svm.SVC(kernel=kernel, decision_function_shape='ovo')
    clf.fit(train_data, train_labels)

    train_predictions = []
    train_predictions = clf.predict(train_data)

    train_accuracy = calculate_accuracy(train_predictions, train_labels)
    train_precision, train_recall, train_f1 = calculate_metrics(train_predictions, train_labels)

    # print "Training Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (
    #     train_accuracy, train_precision, train_recall, train_f1)

    test_predictions = []
    test_predictions = clf.predict(test_data)
    accuracy = calculate_accuracy(test_predictions, test_labels)
    precision, recall, f1 = calculate_metrics(test_predictions, test_labels)
    # print "Testing Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (
    #     accuracy, precision, recall, f1)

    # accuracy = train_accuracy
    # precision = train_precision
    # recall = train_recall
    # f1 = train_f1

    return accuracy, precision, recall, f1


def normal_classification(X_data, Y_data):

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.125)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        # print "Fold%d -> Number of training samples: %d | Number of testing "\
        #     "samples: %d" % (fold, len(train_indices), len(test_indices))
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        accumulated_metrics.append(SVM(train_data, train_labels, test_data, test_labels, svm_kernel))
        # print(train_data[:, i].shape)
        fold += 1

    final_metrics = zip(*accumulated_metrics)
    final_accuracy = sum(final_metrics[0]) / len(final_metrics[0])
    final_precision = sum(final_metrics[1]) / len(final_metrics[1])
    final_recall = sum(final_metrics[2]) / len(final_metrics[2])
    final_f1 = sum(final_metrics[3]) / len(final_metrics[3])

    print "Testing Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (
        final_accuracy, final_precision, final_recall, final_f1)



def find_best_contributors(X_data, Y_data):
    accuracy = {}
    for i in range(0, X_data.shape[1]):
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.125)   # Do not change this split size
        accumulated_metrics = []
        fold = 1
        for train_indices, test_indices in sss.split(X_data, Y_data):
            # print "Fold%d -> Number of training samples: %d | Number of testing "\
            #     "samples: %d" % (fold, len(train_indices), len(test_indices))
            train_data, test_data = X_data[train_indices], X_data[test_indices]
            train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
            train_data[:, [j for j in range(1, train_data.shape[1]) if j != i]] = 0
            test_data[:, [j for j in range(1, train_data.shape[1]) if j != i]] = 0
            accumulated_metrics.append(SVM(train_data, train_labels, test_data, test_labels, svm_kernel))
            # print(train_data[:, i].shape)
            fold += 1

        final_metrics = zip(*accumulated_metrics)
        final_accuracy = sum(final_metrics[0]) / len(final_metrics[0])
        final_precision = sum(final_metrics[1]) / len(final_metrics[1])
        final_recall = sum(final_metrics[2]) / len(final_metrics[2])
        final_f1 = sum(final_metrics[3]) / len(final_metrics[3])
        accuracy[i] = final_accuracy

        print "Testing feature: %d, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (
            i, final_accuracy, final_precision, final_recall, final_f1)

    discriminators = sorted(accuracy, key=accuracy.get)

    return discriminators[0:2]

def classify_on_disciminators(X_data, Y_data, discriminators):

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.125)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        # print "Fold%d -> Number of training samples: %d | Number of testing "\
        #     "samples: %d" % (fold, len(train_indices), len(test_indices))
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        train_data[:, [j for j in range(1, train_data.shape[1]) if j not in discriminators]] = 0
        test_data[:, [j for j in range(1, train_data.shape[1]) if j not in discriminators]] = 0
        accumulated_metrics.append(SVM(train_data, train_labels, test_data, test_labels, svm_kernel))
        # print(train_data[:, i].shape)
        fold += 1

    final_metrics = zip(*accumulated_metrics)
    final_accuracy = sum(final_metrics[0]) / len(final_metrics[0])
    final_precision = sum(final_metrics[1]) / len(final_metrics[1])
    final_recall = sum(final_metrics[2]) / len(final_metrics[2])
    final_f1 = sum(final_metrics[3]) / len(final_metrics[3])

    print "Discriminating features: %d, %d, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (
        discriminators[0], discriminators[1], final_accuracy, final_precision, final_recall, final_f1)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        print "Usage: python letter_classification_svm.py --data_dir='<dataset dir path>'"
        sys.exit()
    else:
        filename = os.path.join(args.data_dir, 'letter_classification_train.data')
        try:
            if os.path.exists(filename):
                print "Using %s as the dataset file" % filename
        except:
            print "%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir)
            sys.exit()

    # Set the value for svm_kernel as required.
    svm_kernel = 'linear'

    X_data, Y_data = get_input_data(filename)
    print X_data.shape, Y_data.shape

    normal_classification(X_data, Y_data)
    discriminators = find_best_contributors(X_data, Y_data)
    classify_on_disciminators(X_data, Y_data, discriminators)
