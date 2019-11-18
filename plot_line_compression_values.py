#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:14:10 2019

@author: zhouhonglu
"""

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
    file = "/Users/zhouhonglu/Downloads/CompDist.csv"
    save_pathfile = "/Users/zhouhonglu/Downloads/CompDist_"

    df = pd.read_csv(file)

    comparisons = {
            'C1': 'CR 1',
            'C1.25': 'CR 1.25',
            'C1.5': 'CR 1.5',
            'C1.75': 'CR 1.75',
            'C2': 'CR 2'
            }


    colors_all = {

            'C1': 'g'
            ,
            'C1.25': 'lime',
            'C1.5': 'b',
            'C1.75': 'm',
            'C2': 'r',
            'M6': 'lime',
            'M8': 'b'
            }


    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 10
    # plt.rcParams["axes.labelweight"] = "bold"

    figsize = (7, 2.5)

    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()

    x = np.arange(1, 7+1)
    for cr in comparisons:
        plt.plot(x,df[cr], label=comparisons[cr], color=colors_all[cr], alpha=0.8)


    # general layout
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    # plt.xticks(xlabel_pos, comparisons)
    plt.xlabel('Compression Value', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)

    plt.legend(loc='upper right', fontsize=12)
    plt.title('Histogram of Compression Values in the Compression Channels', fontsize=12)
    plt.tight_layout()
#    plt.show()
#    os._exit(0)
    plt.savefig(save_pathfile+ '_M8_M10to13.png')
    print("{} saved!".format(save_pathfile + '_M8M10_KL.png'))

   # pdb.set_trace()
