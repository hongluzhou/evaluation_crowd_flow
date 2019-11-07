#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 23:38:26 2019

@author: zhouhonglu
"""

import os
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
https://matplotlib.org/examples/color/named_colors.html
"""


if __name__ == '__main__':
    file = "/Users/zhouhonglu/Downloads/M8vsM10.csv"
    save_pathfile = "/Users/zhouhonglu/Downloads/M8vsM10_"

    df = pd.read_csv(file)

    data = [df.columns.values.tolist()] + df.values.tolist()

    comparisons = ['CR 1', 'CR 1.25', 'CR 1.5', 'CR 1.75', 'CR 2']


    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"

    barWidth = 0.1
    interval = 0.05
    figsize = (8, 3)



    """
    KL
    """
    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()

    M8_y = []
    M8_errors = []
    M8_x = np.arange(len(comparisons))
    xlabel_pos = M8_x
    for i in range(len(comparisons)):
        # KL mean
        M8_y.append(data[3][i+1])
        # KL STD
        M8_errors.append(data[4][i+1])
    label='M8'
    plt.errorbar(M8_x, M8_y, yerr=M8_errors, label=label, color='b', uplims=False, lolims=False)


    M10_y = []
    M10_errors = []
    M10_x = np.arange(len(comparisons))
    xlabel_pos = M10_x
    for i in range(len(comparisons)):
        # KL mean
        M10_y.append(data[3][i+1+len(comparisons)])
        # KL STD
        M10_errors.append(data[4][i+1+len(comparisons)])
    label='M10'
    plt.errorbar(M10_x, M10_y, yerr=M10_errors, label=label, color='r', uplims=False, lolims=False)


    # general layout
    ax.yaxis.grid(True)
    plt.xticks(xlabel_pos, comparisons)
    plt.ylabel('KL Divergence')
    plt.legend(loc='upper right')
    plt.title('Quantitative Evaluation of the Compression Experiment')
    plt.tight_layout()

    plt.savefig(save_pathfile+ '_M8M10_KL.png')
    # plt.show()
    print("{} saved!".format(save_pathfile + '_M8M10_KL.png'))


    """
    MAE
    """
    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()

    M8_y = []
    M8_errors = []
    M8_x = np.arange(len(comparisons))
    xlabel_pos = M8_x
    for i in range(len(comparisons)):
        # MAE mean
        M8_y.append(data[1][i+1])
        # MAE STD
        M8_errors.append(data[2][i+1])
    label='M8'
    plt.errorbar(M8_x, M8_y, yerr=M8_errors, label=label, color='b', uplims=False, lolims=False, alpha=0.9)

    M10_y = []
    M10_errors = []
    M10_x = np.arange(len(comparisons))
    xlabel_pos = M10_x
    for i in range(len(comparisons)):
        # MAE mean
        M10_y.append(data[1][i+1+len(comparisons)])
        # MAE STD
        M10_errors.append(data[2][i+1+len(comparisons)])
    label='M10'
    plt.errorbar(M10_x, M10_y, yerr=M10_errors, label=label, color='r', uplims=False, lolims=False, alpha=0.6)


    # general layout
    ax.yaxis.grid(True)
    plt.xticks(xlabel_pos, comparisons)
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='upper right')  #
    plt.title('Quantitative Evaluation of the Compression Experiment')
    plt.tight_layout()

    plt.savefig(save_pathfile+ '_M8M10_MAE.png')
    # plt.show()
    print("{} saved!".format(save_pathfile + '_M8M10_MAE.png'))




    # pdb.set_trace()
