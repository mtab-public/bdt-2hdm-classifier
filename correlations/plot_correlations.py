#!/usr/bin/python3

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["figure.figsize"] = (8, 6)

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
systematic_error = 0.10


#################################################
# INPUT/OUTPUT DATA
#################################################

# features and labels
features_for_training = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
feature_label = [
    r'$n_\mathrm{jets}$',
    r'$n_\mathrm{b\ jets}$',
    r'$p_T^1}$',
    r'$p_T^2}$',
    r'$p_T^3}$',
    r'$R_{12}^2}$',
    r'$R_{13}^2}$',
    r'$R_{23}^2}$',
    r'$\eta_1}$',
    r'$\eta_2}$',
    r'$\eta_3$',
    r'$R_{l1}^2}$',
    r'$R_{l2}^2}$',
    r'$R_{l3}^2}$',
    r'$p_T^l}$',
    r'$\eta_l}$',
    r'$E_T^\mathrm{miss}$',
    r'$H_T$'
]

# signal region
signal_region = "3b_1l/"


#################################################
# background

# folder to output event data
output_folder_data = "../data/" + signal_region

# background data
background_process = ["ttbarjs_5FS", "wt_jets", "tj_plus_1jet", "ttbarh", "ttbarZ"]
num_files_bg = {"ttbarjs_5FS": 10000, "wt_jets": 500, "tj_plus_1jet": 1000, "ttbarh": 1, "ttbarZ": 1}
K_factors = {"ttbarjs_5FS": 1.6, "wt_jets": 1.18, "tj_plus_1jet": 1, "ttbarh": 1, "ttbarZ": 1}


#################################################
# signal

# signal data
signal_process_folder = "jshpm_tb_5FS/"

signal_folder = "signal/" + signal_process_folder # + signal_region

signal_process = [command_line_args[1]]
num_files_sg = 100


#################################################
# MAIN PROGRAM
#################################################

n_features = 18 + 4

# number of test events
n_test_bg = {}
n_test_sg = {}

# import background
X_bg = {}
temp = np.fromfile(output_folder_data + background_process[0] + ".dat")
X_bg[background_process[0]] = temp.reshape(int(len(temp)/n_features), n_features)

X_bg[background_process[0]] = np.asarray(
        [ X_bg[background_process[0]][ii] for ii in range(0, len(X_bg[background_process[0]]))
            if ( X_bg[background_process[0]][ii][20] > ET_miss_cut and X_bg[background_process[0]][ii][21] > HT_cut ) ]
    )

n_test_bg[background_process[0]] = int(np.ceil(np.sum(X_bg[background_process[0]][:, 0])/num_files_bg[background_process[0]]*1e12*Lumi)*K_factors[background_process[0]])

print('Available background events: ', len(X_bg[background_process[0]]))
print('Plotting ' + str(n_test_bg[background_process[0]]) + ' background events\n')

# import signal
X_sg = {}
temp = np.fromfile(output_folder_data + signal_process_folder + signal_process[0] + ".dat")
X_sg[signal_process[0]] = temp.reshape(int(len(temp)/n_features), n_features)

X_sg[signal_process[0]] = np.asarray(
        [ X_sg[signal_process[0]][ii] for ii in range(0, len(X_sg[signal_process[0]]))
            if ( X_sg[signal_process[0]][ii][20] > ET_miss_cut and X_sg[signal_process[0]][ii][21] > HT_cut ) ]
    )

n_test_sg[signal_process[0]] = int(np.sum(X_sg[signal_process[0]][:, 0])/num_files_sg*1e12*Lumi)

print('Available signal events: ', len(X_sg[signal_process[0]]))
print('Plotting ' + str(n_test_sg[signal_process[0]]) + ' signal events\n')

#for i in range(0, len(feature_label)):
#    for j in range(i + 1, len(feature_label)):
#
#        plot_features = [features_for_training[i], features_for_training[j]]
#
#        features_ij_bg = X_bg[background_process[0]][:n_test_bg[background_process[0]]][:, plot_features]
#        features_ij_sg = X_sg[signal_process[0]][:n_test_sg[signal_process[0]]][:, plot_features]
#
##        plt.gcf().set_size_inches(18.5, 10.5)
#        
#        plt.scatter(features_ij_bg[:, 0], features_ij_bg[:, 1])
#        plt.scatter(features_ij_sg[:, 0], features_ij_sg[:, 1])
#
#        plt.xlabel(feature_label[i], fontsize = 18)
#        plt.ylabel(feature_label[j], fontsize = 18)
#
#        plt.xticks(fontname = "Latin Modern Roman", fontsize = 18)
#        plt.yticks(fontname = "Latin Modern Roman", fontsize = 18)
#
#        plt.grid()
#
#        plt.savefig('scatter_' + str(i) + '_' + str(j) + '.png', dpi=300)
#
#        plt.clf()

# plot number of jet distributions

num_jets_bg = [int(i) for i in X_bg[background_process[0]][:n_test_bg[background_process[0]]][:, 1]]
num_jets_sg = [int(i) for i in X_sg[signal_process[0]][:n_test_sg[signal_process[0]]][:, 1]]

#        plt.gcf().set_size_inches(18.5, 10.5)

bg_array = [[0 for x in range(2)] for y in range(20)]
sg_array = [[0 for x in range(2)] for y in range(20)]

for i in range(20):
    bg_array[i][0] = i
    bg_array[i][1] = num_jets_bg.count(i)

    sg_array[i][0] = i
    sg_array[i][1] = num_jets_sg.count(i)

sg_array = np.asarray(sg_array)
bg_array = np.asarray(bg_array)

plt.scatter(bg_array[:, 0], bg_array[:, 1])
plt.scatter(sg_array[:, 0], sg_array[:, 1])

plt.yscale('log', basey=10)

plt.xlim([3, 17])
plt.ylim([1, 100000])

plt.xticks(np.arange(3, 18, 1))

plt.ylabel( r'$n_\mathrm{events}$', fontsize = 18)
plt.xlabel(feature_label[0], fontsize = 18)

plt.xticks(fontname = "Latin Modern Roman", fontsize = 18)
plt.yticks(fontname = "Latin Modern Roman", fontsize = 18)

plt.grid()

plt.savefig('scatter_' + str(0) + '_' + str(0) + '.png', dpi=300)

plt.clf()