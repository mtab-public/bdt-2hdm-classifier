#!/usr/bin/python3
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
# Import test datasets
from sklearn import datasets
from sklearn.datasets import make_gaussian_quantiles
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#################################################
# HELP
#################################################
if len(sys.argv) != 2:
    print('\n Usage ' + sys.argv[0] +  ' [signal mass] \n')
    sys.exit()

command_line_args = sys.argv

#################################################
# CUTS
#################################################
ET_miss_cut = 35
HT_cut = 350
Lumi = 140
systematic_error = 0.05

#################################################
# INPUT/OUTPUT DATA
#################################################

# path to event files
path = "./"

# signal region
signal_region = "3b_1l/"


#################################################
# background

# folder to output background event data
# since same for all signals
output_folder_data = "data/" + signal_region
os.makedirs(output_folder_data, exist_ok=True)

# background data
background_folder = "background_" + signal_region
background_process = ["ttbarjs_5FS", "wt_jets", "tj_plus_1jet", "ttbarh", "ttbarZ"]

num_files_bg = {"ttbarjs_5FS": 10000, "wt_jets": 500, "tj_plus_1jet": 1000, "ttbarh": 1, "ttbarZ": 1}
K_factors = {"ttbarjs_5FS": 1.6, "wt_jets": 1.18, "tj_plus_1jet": 1, "ttbarh": 1, "ttbarZ": 1}


#################################################
# signal

# signal data
signal_process_folder = "jshpm_tb_5FS/"

# folder to output background event data
# since same for all signals
os.makedirs(output_folder_data + signal_process_folder, exist_ok=True)

signal_folder = "signal/" + signal_process_folder # + signal_region
signal_process = [command_line_args[1]]
num_files_sg = 100


#################################################
# bdt

# output model name
filename = "models/" + signal_region + signal_process_folder + "mhc_" + signal_process[0] + ".sav"
os.makedirs("models/" + signal_region + signal_process_folder, exist_ok=True)

# number of training events per process
n_train_bg = 100000
print("\nTraining on " + str(n_train_bg) +  " background events")

n_train_sg = 10000
print("\nTraining on " + str(n_train_sg) + " signal events")

# number of test events
n_test_bg = {}
n_test_sg = {}

# number of features
n_features = 18 + 4

features_for_training = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

#################################################
# MAIN PROGRAM
#################################################

# preprocess background data
X_bg = {}
y_bg = {}
weight_bg = {}
for process in background_process:
    if os.path.isfile(output_folder_data + process + ".dat"):
        print("\nImporting existing data")
        temp = np.fromfile(output_folder_data + process + ".dat")
        X_bg[process] = temp.reshape(int(len(temp)/n_features), n_features)

    elif os.path.isfile(path + background_folder + process + "/" + process + "_" + str(1) + ".txt"):
        print("\nRaw data found, processing...")
        temp = []

        for i in range(0, num_files_bg[process]):
            temp.append(np.loadtxt(path + background_folder + process + "/" + process + "_" + str(i + 1) + ".txt"))

        X_bg[process] = np.concatenate(temp)
        X_bg[process].tofile(output_folder_data + process + ".dat")

    else:
        print('\nBackground process ' + process + ' not found\n')
        sys.exit()

    X_bg[process] = np.asarray(
        [ X_bg[process][ii] for ii in range(0, len(X_bg[process]))
            if ( X_bg[process][ii][20] > ET_miss_cut and X_bg[process][ii][21] > HT_cut ) ]
    )

    y_bg[process] = np.full(len(X_bg[process]), 0)

    weight_bg[process] = X_bg[process][:, 0]

    n_test_bg[process] = int(np.ceil(np.sum(X_bg[process][:, 0])/num_files_bg[process]*1e12*Lumi)*K_factors[process])

    print("Background events " + process + ": ", len(X_bg[process]))
    print("Expected bg events " + process + ": ", n_test_bg[process])


# preprocess signal data
X_sg = {}
y_sg = {}
weight_sg = {}
for process in signal_process:
    if os.path.isfile(output_folder_data + signal_process_folder + process + ".dat"):
        print("\nImporting existing data")
        temp = np.fromfile(output_folder_data + signal_process_folder + process + ".dat")
        X_sg[process] = temp.reshape(int(len(temp)/n_features),n_features)

    elif os.path.isfile(path + signal_folder + process + "_" + str(1) + ".txt"):
        print("\nRaw data found, processing...")
        temp = []

        for i in range(0, num_files_sg):
            temp.append(np.loadtxt(path + signal_folder + process + "_" + str(i + 1) + ".txt"))

        X_sg[process] = np.concatenate(temp)
        X_sg[process].tofile(output_folder_data + signal_process_folder + process + ".dat")

    else:
        print('\nSignal process ' + process + ' not found\n')
        sys.exit()

    X_sg[process] = np.asarray(
        [ X_sg[process][ii] for ii in range(0, len(X_sg[process]))
            if ( X_sg[process][ii][20] > ET_miss_cut and X_sg[process][ii][21] > HT_cut ) ]
    )

    y_sg[process] = np.full(len(X_sg[process]), 1)

    weight_sg[process] = X_sg[process][:, 0]

    n_test_sg[process] = int(np.sum(X_sg[process][:, 0])/num_files_sg*1e12*Lumi)

    print("Signal events " + process + ": ", len(X_sg[process]))
    print("Expected sg events " + process + ": ", n_test_sg[process])

print()

# get significance on full features
# training dataset
X_train = np.concatenate( (
    np.abs(X_bg[background_process[0]][0:n_train_bg][:, features_for_training]),
    np.abs(X_sg[signal_process[0]][0:n_train_sg][:, features_for_training])
) )
y_train = np.concatenate( (
    y_bg[background_process[0]][0:n_train_bg],
    y_sg[signal_process[0]][0:n_train_sg]
) )
w_train = np.concatenate( (
    weight_bg[background_process[0]][0:n_train_bg],
    weight_sg[signal_process[0]][0:n_train_sg]
) )

# test dataset
X_test = np.concatenate( (
    np.abs(X_bg[background_process[0]][n_train_bg:n_train_bg + n_test_bg[background_process[0]]][:, features_for_training]),
    np.abs(X_bg[background_process[1]][0:n_test_bg[background_process[1]]][:, features_for_training]),
    np.abs(X_bg[background_process[2]][0:n_test_bg[background_process[2]]][:, features_for_training]),
    np.abs(X_bg[background_process[3]][0:n_test_bg[background_process[3]]][:, features_for_training]),
    np.abs(X_bg[background_process[4]][0:n_test_bg[background_process[4]]][:, features_for_training]),
    np.abs(X_sg[signal_process[0]][n_train_sg:n_train_sg + n_test_sg[signal_process[0]]][:, features_for_training])
) )

y_test = np.concatenate( (
    np.full(n_test_bg[background_process[0]], 0),
    np.full(n_test_bg[background_process[1]], 0),
    np.full(n_test_bg[background_process[2]], 0),
    np.full(n_test_bg[background_process[3]], 0),
    np.full(n_test_bg[background_process[4]], 0),
    np.full(n_test_sg[signal_process[0]], 1)
) )

# create adaboost classifer object
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=14, min_weight_fraction_leaf=0.01), n_estimators=10, learning_rate=0.4)

# train adaboost classifer
bdt.fit(X_train, y_train)#, w_train)

# predict the response for test dataset
y_pred = bdt.predict(X_test)
y_pred_train = bdt.predict(X_train)

# model accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))

twoclass_output = bdt.decision_function(X_test)
plot_range = (twoclass_output.min(), twoclass_output.max())
class_names = "BS"
n_events = {}
bins = {}
plt.subplot(211)
for i, n in zip(range(2), class_names):
    (n_events[i], bins[i], temp) = plt.hist(twoclass_output[y_test == i],
                                            bins=30,
                                            range=plot_range,
                                            label=n,
                                            alpha=.5,
                                            edgecolor='k')

# Calculate and plot significance
n_sg = n_events[1]
n_bg = n_events[0]

bins_sq = bins[1]
bins_bg = bins[0]

significance = np.zeros((len(n_sg), 2))
for i in range(len(n_sg)):
    n_s = np.sum(n_sg[i:])
    n_b = np.sum(n_bg[i:])
    significance[i] = [(bins_bg[i] + bins_bg[i + 1])/2, n_s/np.sqrt(n_s + n_b + pow(n_b*systematic_error, 2))]

print("\nmax sig : ", max(significance[:, 1]), "\n")

sys.stdout.flush()

plt.clf()
plt.cla()

# delete single features and analyse model
for idx_feature in range(0, len(features_for_training)):

    reduced_features = features_for_training[:]

    del reduced_features[idx_feature]
    print( "Features included: ", reduced_features)

    # training dataset
    X_train = np.concatenate( (
        np.abs(X_bg[background_process[0]][0:n_train_bg][:, reduced_features]),
        np.abs(X_sg[signal_process[0]][0:n_train_sg][:, reduced_features])
    ) )
    y_train = np.concatenate( (
        y_bg[background_process[0]][0:n_train_bg],
        y_sg[signal_process[0]][0:n_train_sg]
    ) )
    w_train = np.concatenate( (
        weight_bg[background_process[0]][0:n_train_bg],
        weight_sg[signal_process[0]][0:n_train_sg]
    ) )

    # test dataset
    X_test = np.concatenate( (
        np.abs(X_bg[background_process[0]][n_train_bg:n_train_bg + n_test_bg[background_process[0]]][:, reduced_features]),
        np.abs(X_bg[background_process[1]][0:n_test_bg[background_process[1]]][:, reduced_features]),
        np.abs(X_bg[background_process[2]][0:n_test_bg[background_process[2]]][:, reduced_features]),
        np.abs(X_bg[background_process[3]][0:n_test_bg[background_process[3]]][:, reduced_features]),
        np.abs(X_bg[background_process[4]][0:n_test_bg[background_process[4]]][:, reduced_features]),
        np.abs(X_sg[signal_process[0]][n_train_sg:n_train_sg + n_test_sg[signal_process[0]]][:, reduced_features])
    ) )

    y_test = np.concatenate( (
        np.full(n_test_bg[background_process[0]], 0),
        np.full(n_test_bg[background_process[1]], 0),
        np.full(n_test_bg[background_process[2]], 0),
        np.full(n_test_bg[background_process[3]], 0),
        np.full(n_test_bg[background_process[4]], 0),
        np.full(n_test_sg[signal_process[0]], 1)
    ) )

    # create adaboost classifer object
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=14, min_weight_fraction_leaf=0.01), n_estimators=10, learning_rate=0.4)

    # train adaboost classifer
    bdt.fit(X_train, y_train)#, w_train)

    # predict the response for test dataset
    y_pred = bdt.predict(X_test)
    y_pred_train = bdt.predict(X_train)

    # model accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))

    twoclass_output = bdt.decision_function(X_test)
    plot_range = (twoclass_output.min(), twoclass_output.max())
    class_names = "BS"
    n_events = {}
    bins = {}
    plt.subplot(211)
    for i, n in zip(range(2), class_names):
        (n_events[i], bins[i], temp) = plt.hist(twoclass_output[y_test == i],
                                                bins=30,
                                                range=plot_range,
                                                label=n,
                                                alpha=.5,
                                                edgecolor='k')

    # Calculate and plot significance
    n_sg = n_events[1]
    n_bg = n_events[0]

    bins_sq = bins[1]
    bins_bg = bins[0]

    significance = np.zeros((len(n_sg), 2))
    for i in range(len(n_sg)):
        n_s = np.sum(n_sg[i:])
        n_b = np.sum(n_bg[i:])
        significance[i] = [(bins_bg[i] + bins_bg[i + 1])/2, n_s/np.sqrt(n_s + n_b + pow(n_b*systematic_error, 2))]

    print("\nmax sig for ", idx_feature, " : ", max(significance[:, 1]), "\n")

    sys.stdout.flush()

    plt.clf()
    plt.cla()


# delete pair of features and analyse model
for idx_feature in range(0, len(features_for_training)):
    for idx_feature2 in range(idx_feature + 1, len(features_for_training)):

        reduced_features = features_for_training[:]

        del reduced_features[idx_feature2]
        del reduced_features[idx_feature]

        print( "Features included: ", reduced_features)

        # training dataset
        X_train = np.concatenate( (
            np.abs(X_bg[background_process[0]][0:n_train_bg][:, reduced_features]),
            np.abs(X_sg[signal_process[0]][0:n_train_sg][:, reduced_features])
        ) )
        y_train = np.concatenate( (
            y_bg[background_process[0]][0:n_train_bg],
            y_sg[signal_process[0]][0:n_train_sg]
        ) )
        w_train = np.concatenate( (
            weight_bg[background_process[0]][0:n_train_bg],
            weight_sg[signal_process[0]][0:n_train_sg]
        ) )

        # test dataset
        X_test = np.concatenate( (
            np.abs(X_bg[background_process[0]][n_train_bg:n_train_bg + n_test_bg[background_process[0]]][:, reduced_features]),
            np.abs(X_bg[background_process[1]][0:n_test_bg[background_process[1]]][:, reduced_features]),
            np.abs(X_bg[background_process[2]][0:n_test_bg[background_process[2]]][:, reduced_features]),
            np.abs(X_bg[background_process[3]][0:n_test_bg[background_process[3]]][:, reduced_features]),
            np.abs(X_bg[background_process[4]][0:n_test_bg[background_process[4]]][:, reduced_features]),
            np.abs(X_sg[signal_process[0]][n_train_sg:n_train_sg + n_test_sg[signal_process[0]]][:, reduced_features])
        ) )

        y_test = np.concatenate( (
            np.full(n_test_bg[background_process[0]], 0),
            np.full(n_test_bg[background_process[1]], 0),
            np.full(n_test_bg[background_process[2]], 0),
            np.full(n_test_bg[background_process[3]], 0),
            np.full(n_test_bg[background_process[4]], 0),
            np.full(n_test_sg[signal_process[0]], 1)
        ) )

        # create adaboost classifer object
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=14, min_weight_fraction_leaf=0.01), n_estimators=10, learning_rate=0.4)

        # train adaboost classifer
        bdt.fit(X_train, y_train)#, w_train)

        # predict the response for test dataset
        y_pred = bdt.predict(X_test)
        y_pred_train = bdt.predict(X_train)

        # model accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))

        twoclass_output = bdt.decision_function(X_test)
        plot_range = (twoclass_output.min(), twoclass_output.max())
        class_names = "BS"
        n_events = {}
        bins = {}
        plt.subplot(211)
        for i, n in zip(range(2), class_names):
            (n_events[i], bins[i], temp) = plt.hist(twoclass_output[y_test == i],
                                                    bins=30,
                                                    range=plot_range,
                                                    label=n,
                                                    alpha=.5,
                                                    edgecolor='k')

        # Calculate and plot significance
        n_sg = n_events[1]
        n_bg = n_events[0]

        bins_sq = bins[1]
        bins_bg = bins[0]

        significance = np.zeros((len(n_sg), 2))
        for i in range(len(n_sg)):
            n_s = np.sum(n_sg[i:])
            n_b = np.sum(n_bg[i:])
            significance[i] = [(bins_bg[i] + bins_bg[i + 1])/2, n_s/np.sqrt(n_s + n_b + pow(n_b*systematic_error, 2))]

        print("\nmax sig for ", idx_feature, " ", idx_feature2 , " : ", max(significance[:, 1]), "\n")

        sys.stdout.flush()

        plt.clf()
        plt.cla()
