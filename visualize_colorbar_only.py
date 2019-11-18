#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:17:29 2019

@author: zhouhonglu

https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors

"""
import os
import pdb
import cv2
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imsave
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def view_colormap(cmapstr):
    cmap = plt.cm.get_cmap(cmapstr)
    colors = cmap(np.arange(cmap.N))

    # pdb.set_trace()
    plt.imshow([colors], extent=[0, 10, 0, 1])
    plt.axis('off')
    plt.savefig(cmapstr + '.png', bbox_inches='tight')

view_colormap('Blues')


#
#output = np.ones((224, 224, 3))
#output[0, 0, :] = [-1, -1, -1]
#plt.figure()
#ax = plt.gca()
#im = ax.imshow(output, cmap='jet', interpolation='nearest')
## create an axes on the right side of ax. The width of cax will be 5%
## of ax and the padding between cax and ax will be fixed at 0.05 inch.
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = plt.colorbar(im, cax=cax)
#cbar.ax.tick_params(labelsize=10)
#ax.axis('off')
## plt.tight_layout()
#plt.show()
## os.exit(0)
#plt.savefig(save_path, bbox_inches='tight', dpi=1000)
#
#crop_image_tight(save_path)
#
#shutil.copyfile(save_path, os.path.join(config['save_path'], "qualitative_results", testcase))
#
#plt.close('all')
