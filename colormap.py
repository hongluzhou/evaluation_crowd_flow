#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:46:48 2019

@author: zhouhonglu
"""
import pdb
import numpy as np
from PIL import Image
from scipy.misc import imsave
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def view_colormap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    pdb.set_trace()
    plt.imshow([colors], extent=[0, 10, 0, 1])
    plt.axis('off')
    plt.savefig('color.png', bbox_inches='tight')

view_colormap('jet')