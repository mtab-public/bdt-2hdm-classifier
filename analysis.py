#!/usr/bin/python3
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import metrics
#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn import tree

#################################################
# HELP
#################################################
if len(sys.argv) != 2:
    print('\n Usage ' + sys.argv[0] +  ' [signal mass] \n')
    sys.exit()

command_line_args = sys.argv

#################################################
# CUTS etc
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

# folder to input event data
folder_data = "data/" + signal_region


#################################################
# background

# background data
background_process = ["ttbarjs_5FS", "wt_jets", "tj_plus_1jet", "ttbarh", "ttbarZ"]

# number raw files used for training
num_files_bg = {"ttbarjs_5FS": 10000, "wt_jets": 500, "tj_plus_1jet": 1000, "ttbarh": 1, "ttbarZ": 1}
K_factors = {"ttbarjs_5FS": 1.6, "wt_jets": 1.18, "tj_plus_1jet": 1, "ttbarh": 1, "ttbarZ": 1}


#################################################
# signal

# signal data
signal_process_folder = "jshpm_tb_5FS/"

signal_process = [command_line_args[1]]

num_files_sg = 100


#################################################
# plots

# folder to store plots
output_plots = "plots/" + signal_region + signal_process_folder
os.makedirs(output_plots, exist_ok=True)


#################################################
# bdt

# output model name
filename = "mhc_" + signal_process[0] + ".sav"
#filename = "mhc_300_BP1.sav"
bdt = pickle.load(open("models/" + signal_region + signal_process_folder + filename, 'rb'))

# number of training events per process
n_train_bg = 100000
n_train_sg = 10000
# dics to save estimated number of test events @{Lumi}fb^-1 rescaled by K factor
n_test_bg = {}
n_test_sg = {}

# number of features
n_features = 18 + 4

features_for_training = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

#################################################
# MAIN PROGRAM
#################################################


#################################################
# process data

# preprocess background data
X_bg = {}
y_bg = {}
weight_bg = {}
print()
max_length = 13
for process in background_process:

    temp = np.fromfile(folder_data + process + ".dat")

    X_bg[process] = temp.reshape(int(len(temp)/n_features), n_features)
    
    X_bg[process] = np.asarray(
        [ X_bg[process][ii] for ii in range(0, len(X_bg[process]))
            if ( X_bg[process][ii][20] > ET_miss_cut and X_bg[process][ii][21] > HT_cut ) ]
    )

    n_test_bg[process] = int(np.ceil(np.sum(X_bg[process][:, 0])/num_files_bg[process]*1e12*Lumi)*K_factors[process])

    y_bg[process] = np.full(len(X_bg[process]), 0)

    weight_bg[process] = X_bg[process][:, 0]

    print("""Background events {process:<{max_length}}""".format(**locals()), len(X_bg[process]))
    print("""Expected b events {process:<{max_length}}""".format(**locals()), n_test_bg[process])

print()

# preprocess signal data
X_sg = {}
y_sg = {}
weight_sg = {}
for process in signal_process:
    
    temp = np.fromfile(folder_data + signal_process_folder + process + ".dat")

    X_sg[process] = temp.reshape(int(len(temp)/n_features), n_features)

    X_sg[process] = np.asarray(
        [ X_sg[process][ii] for ii in range(0, len(X_sg[process]))
            if ( X_sg[process][ii][20] > ET_miss_cut and X_sg[process][ii][21] > HT_cut ) ]
    )

    n_test_sg[process] = int(np.sum(X_sg[process][:, 0])/num_files_sg*1e12*Lumi)

    y_sg[process] = np.full(len(X_sg[process]), 1)

    weight_sg[process] = X_sg[process][:, 0]

    print("Signal events " + process + ": ", len(X_sg[process]))
    print("Number of expected signal events " + process + ": ", n_test_sg[process])

print()


#################################################
# train BDT

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

# Predict the response for test dataset
y_pred_test = bdt.predict(X_test)
y_pred_train = bdt.predict(X_train)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))


#################################################
# confusion matrix

conf_mat = confusion_matrix(y_test, y_pred_test)
print(conf_mat)


#################################################
# determine bdt score
#plt.rcParams["font.family"] = ["Latin Modern Roman", "serif"]
#plt.rcParams["font.size"] = 14
#plt.rcParams["font.weight"] = "bold"

# Plot the two-class decision scores
plt.figure(0)
twoclass_output = bdt.decision_function(X_test)

plot_range = (twoclass_output.min(), twoclass_output.max())

#plot_colors = "br"
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

plt.yscale('log', basey=10)
#x1, x2, y1, y2 = plt.axis()
#plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right', prop={"family":"Latin Modern Roman"})
plt.ylabel('Events', fontname = "Latin Modern Roman", fontsize = 14)
plt.yticks(fontname = "Latin Modern Roman", fontsize = 14)
plt.xlabel('BDT Score', fontname = "Latin Modern Roman", fontsize = 14)
plt.xticks(fontname = "Latin Modern Roman", fontsize = 14)
#plt.grid()

n_sg = n_events[1]
n_bg = n_events[0]

bins_sq = bins[1]
bins_bg = bins[0]

significance = np.zeros((len(n_sg), 2))
for i in range(len(n_sg)):
    n_s = np.sum(n_sg[i:])
    n_b = np.sum(n_bg[i:])
    significance[i] = [(bins_bg[i] + bins_bg[i + 1])/2, n_s/np.sqrt(n_s + n_b + pow(n_b*systematic_error, 2))]

idx_max = np.argmax(significance[:, 1])
print('\nNumber of selected signal events: ', np.sum(n_sg[idx_max:]))
print('Number of selected background events: ', np.sum(n_bg[idx_max:]))
print("\n=> max sig: ", significance[idx_max, 1], "\n")
print("BDT cut: ", (bins_bg[idx_max] + bins_bg[idx_max + 1])/2, "\n")

plt.subplot(212)
plt.plot(np.transpose(significance)[0], np.transpose(significance)[1])

plt.ylabel('Significance', fontname = "Latin Modern Roman", fontsize = 14)
plt.yticks(fontname = "Latin Modern Roman", fontsize = 14)
plt.xlabel('BDT Score', fontname = "Latin Modern Roman", fontsize = 14)
plt.xticks(fontname = "Latin Modern Roman", fontsize = 14)
plt.grid()

plt.tight_layout()

plt.savefig(output_plots + "scores_and_significance_" + filename + ".pdf")


#################################################
# ROC curve

# Plot ROC curve
plt.figure(1)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(y_test[:], twoclass_output[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), twoclass_output.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontname = "Latin Modern Roman", fontsize = 14)
plt.xticks(fontname = "Latin Modern Roman", fontsize = 14)
plt.ylabel('True Positive Rate', fontname = "Latin Modern Roman", fontsize = 14)
plt.yticks(fontname = "Latin Modern Roman", fontsize = 14)
plt.title('Receiver operating characteristic', fontname = "Latin Modern Roman", fontsize = 14)
plt.legend(loc="lower right", prop={"family":"Latin Modern Roman"})
plt.grid()

plt.savefig(output_plots + "roc_" + filename + ".pdf")


#################################################
# training information

# Plot fit scores
plt.figure(2)
scores = {'train' : [], 'test' : []}
for y_predict_train, y_predict_test in zip(bdt.staged_predict(X_train),
                                           bdt.staged_predict(X_test)):
    scores['train'].append(metrics.accuracy_score(y_train, y_predict_train))
    scores['test'].append(metrics.accuracy_score(y_test, y_predict_test))

# Plot the results.
n_estimators = range(1, len(scores['train']) + 1)
for key in scores.keys():
    plt.plot(n_estimators, scores[key])
plt.title('Staged Scores')
plt.ylabel('Accuracy')
plt.xlabel('N Estimators')
plt.legend(scores.keys())

plt.savefig(output_plots + "train_scores_" + filename + ".pdf")

exit()

#################################################
# PLOT BDT

# Plot BDT
import graphviz
for idx_tree in range(0, len(bdt.estimators_)):

    dot_data = tree.export_graphviz(bdt.estimators_[idx_tree], out_file=None) 
    graph = graphviz.Source(dot_data)
    graph.render("bdt_plots/bdt_" + str(idx_tree))

# Parameters
plt.figure(3)
n_classes = 2
plot_colors = "ry"

for pairidx, pair in enumerate([[16, 17]]):
    # We only take the two corresponding features

#    bdt2 = DecisionTreeClassifier()
#    bdt2.fit(X, y)

    # Plot the decision boundary
    plt.subplot(1, 2, pairidx + 1)

#    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx, yy = np.meshgrid(
#        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
#    )
#    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

#    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # Plot the training points
    plt.xlim([0,1500])
    plt.ylim([300,2000])
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_test == i)
        plt.scatter(
            X_test[idx, pair[0]],
            X_test[idx, pair[1]],
            c=color
        )
        

plt.subplot(1,2,2)
plt.xlim([0,1500])
plt.ylim([300, 2000])
for i, color in zip(range(n_classes), plot_colors):
    idx_sg_bg = np.where(y_test == i)
    idx_cut = np.where(twoclass_output > 0.17)
    idx = np.intersect1d(idx_sg_bg, idx_cut)
    plt.scatter(
        X_test[idx, 16],
        X_test[idx, 17],
        c=color
    )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
plt.savefig("test_2.png", dpi=300)
