#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:38:18 2019

@author: zhouhonglu
"""
import os
import pdb
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imsave
import matplotlib.pyplot as plt
import shutil


def contat_images_anysize_vertical(list_im):
    """
    https://stackoverflow.com/a/30228789/10509909
    """
    imgs = [Image.open(i) for i in list_im]
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    imgs_comb = Image.fromarray(imgs_comb)
    return imgs_comb


list_im = ['framework2.jpeg', 'CAGE.jpeg']
imgs_comb = contat_images_anysize_vertical(list_im)
imgs_comb.save('./framework.png')
