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
    file = "/Users/zhouhonglu/Downloads/M1-M6vsM8.csv"
    save_pathfile = "/Users/zhouhonglu/Downloads/M1-M6vsM8_"

    df = pd.read_csv(file)

    data = [df.columns.values.tolist()] + df.values.tolist()

    comparisons = ['LoS A', 'LoS B', 'LoS C', 'LoS D', 'LoS E', 'LoS F']
    # comparisons = ['LoS A']


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

    colors_all = {
            'M1': 'g',
            'M2': 'y',
            'M3': 'c',
            'M4': 'm',
            'M5': 'r',
            'M6': 'lime',
            'M8': 'b'
            }

    # for each comparison
    M8_bars = []
    M8_errors = []
    M8_pos = []
    xlabel_pos = []
    for i in range(len(comparisons)):
        case = comparisons[i]
        # KL mean
        bar = [data[3][i+1]]
        # KL STD
        error = [data[4][i+1]]
        x_pos = [i * (barWidth*2 + interval) + barWidth]
        label = data[0][i+1]
        plt.bar(x_pos, bar, yerr=error, align='center', width=barWidth, color=colors_all[label], edgecolor='black', capsize=7, label=label)
        xlabel_pos.append(x_pos[0] + barWidth/2)

        # KL mean
        M8_bars.append(data[3][i+1+len(comparisons)])
        # KL STD
        M8_errors.append(data[3][i+1+len(comparisons)])
        M8_pos.append(i * (barWidth*2 + interval) + barWidth*2)
    label='M8'
    plt.bar(M8_pos, M8_bars, yerr=M8_errors, align='center', width=barWidth, color=colors_all[label], edgecolor='black', capsize=7, label=label)



    # general layout
    ax.yaxis.grid(True)
    plt.xticks(xlabel_pos, comparisons)
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.title('Quantitative Comparison of Multi-Density Model and Single-Density Model')
    plt.tight_layout()

    plt.savefig(save_pathfile+ '_KL.png')
    # plt.show()
    print("{} saved!".format(save_pathfile + '_KL.png'))


    """
    MAE
    """
    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()

    colors_all = {
            'M1': 'g',
            'M2': 'y',
            'M3': 'c',
            'M4': 'm',
            'M5': 'r',
            'M6': 'lime',
            'M8': 'b'
            }

    # for each comparison
    M8_bars = []
    M8_errors = []
    M8_pos = []
    xlabel_pos = []
    for i in range(len(comparisons)):
        case = comparisons[i]
        # MAE mean
        bar = [data[1][i+1]]
        # MAE STD
        error = [data[2][i+1]]
        x_pos = [i * (barWidth*2 + interval) + barWidth]
        label = data[0][i+1]
        plt.bar(x_pos, bar, yerr=error, align='center', width=barWidth, color=colors_all[label], edgecolor='black', capsize=7, label=label)
        xlabel_pos.append(x_pos[0] + barWidth/2)

        # MAE mean
        M8_bars.append(data[1][i+1+len(comparisons)])
        # MAE STD
        M8_errors.append(data[2][i+1+len(comparisons)])
        M8_pos.append(i * (barWidth*2 + interval) + barWidth*2)
    label='M8'
    plt.bar(M8_pos, M8_bars, yerr=M8_errors, align='center', width=barWidth, color=colors_all[label], edgecolor='black', capsize=7, label=label)



    # general layout
    ax.yaxis.grid(True)
    plt.xticks(xlabel_pos, comparisons)
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Quantitative Comparison of Multi-Density Model and Single-Density Model')
    plt.tight_layout()

    plt.savefig(save_pathfile + '_MAE.png')
    # plt.show()
    print("{} saved!".format(save_pathfile + '_MAE.png'))


    """
    M8 KL
    """
    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()

    M8_y = []
    M8_errors = []
    M8_x = np.arange(len(comparisons))
    xlabel_pos = M8_x
    for i in range(len(comparisons)):
        # KL mean
        M8_y.append(data[3][i+1+len(comparisons)])
        # KL STD
        M8_errors.append(data[4][i+1+len(comparisons)])
    label='M8'
    plt.errorbar(M8_x, M8_y, yerr=M8_errors, label=label, color='b', uplims=True, lolims=True)

    # general layout
    ax.yaxis.grid(True)
    plt.xticks(xlabel_pos, comparisons)
    plt.ylabel('KL Divergence')
    # plt.legend(loc='upper right')
    plt.title('Quantitative Evaluation of he Multi-Density Model M8')
    plt.tight_layout()

    plt.savefig(save_pathfile+ '_M8_KL.png')
    # plt.show()
    print("{} saved!".format(save_pathfile + '_M8_KL.png'))


    """
    M8 MAE
    """
    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    M8_y = []
    M8_errors = []
    M8_x = np.arange(len(comparisons))
    xlabel_pos = M8_x
    for i in range(len(comparisons)):
        # MAE mean
        M8_y.append(data[1][i+1+len(comparisons)])
        # MAE STD
        M8_errors.append(data[2][i+1+len(comparisons)])
    label='M8'
    plt.errorbar(M8_x, M8_y, yerr=M8_errors, label=label, color='b', uplims=True, lolims=True)

    # general layout
    ax.yaxis.grid(True)
    plt.xticks(xlabel_pos, comparisons)
    plt.ylabel('Mean Absolute Error')
    # plt.legend()  # loc='upper right'
    plt.title('Quantitative Evaluation of the Multi-Density Model M8')
    plt.tight_layout()

    plt.savefig(save_pathfile+ '_M8_MAE.png')
    # plt.show()
    print("{} saved!".format(save_pathfile + '_M8_MAE.png'))




    # pdb.set_trace()
