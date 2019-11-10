#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:51:47 2019

@author: zhouhonglu
"""

import os
import pdb
import cv2
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imsave
import matplotlib.pyplot as plt
import shutil


def create_interval_image(interval_length, save_path=False):
    interval_width = 5

    interval = np.zeros((interval_length, interval_width))
    cm = plt.get_cmap('binary')
    interval = cm(interval)
    interval = (interval[:, :, :3] * 100).astype(np.uint8)

    if save_path:
        imsave(os.path.join(save_path, 'interval.png'), interval)
    return interval


def contat_images_anysize_withpadding_horizontal(list_of_img_path, save_path, config, testcase):
    max_shape = (112, 112)
    interval = create_interval_image(interval_length=max_shape[0])

    list_of_images = []
    for i in range(len(list_of_img_path)):
        file = list_of_img_path[i]
        img = Image.open(file)
        img = np.asarray(img.resize(max_shape))
        list_of_images.append(img)
        if i != (len(list_of_img_path) -1):
            list_of_images.append(interval)

    imgs_comb = np.hstack(tuple(list_of_images))

    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(save_path)

    # add colormap lengend on the right
    output = plt.imread(save_path)

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(output, cmap='jet', interpolation='nearest')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=3)
    ax.axis('off')
    # plt.tight_layout()
    # plt.show()
    # os.exit(0)
    plt.savefig(save_path, bbox_inches='tight', dpi=1000)

    crop_image_tight(save_path)

    shutil.copyfile(save_path, os.path.join(config['save_path'], "qualitative_results", testcase))

    plt.close('all')
    return


def contat_images_anysize_withpadding_horizontal_compression(list_of_img_path, save_path, config, testcase):
    max_shape = (112 * config['compression_rate'], 112 * config['compression_rate'])
    interval = create_interval_image(interval_length=max_shape[0])

    list_of_images = []
    for i in range(len(list_of_img_path)):
        file = list_of_img_path[i]
        img = Image.open(file)
        img = np.asarray(img.resize(max_shape))
        list_of_images.append(img)
        if i != (len(list_of_img_path) -1):
            list_of_images.append(interval)

    imgs_comb = np.hstack(tuple(list_of_images))

    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(save_path)

    # add colormap lengend on the right
    output = plt.imread(save_path)

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(output, cmap='jet', interpolation='nearest')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=3)
    ax.axis('off')
    # plt.tight_layout()
    # plt.show()
    # os.exit(0)
    plt.savefig(save_path, bbox_inches='tight', dpi=1000)

    crop_image_tight(save_path)

    shutil.copyfile(save_path, os.path.join(config['save_path'], "qualitative_results", testcase))

    plt.close('all')
    return


def crop_image_tight(save_path):
    img = Image.open(save_path)
    img = np.asarray(img)

    margin = 5

    for r in range(img.shape[0]):
        if np.sum(np.asarray(img)[r,:,:]) != 255 * img.shape[1] * img.shape[2]:
            row_up = r - margin
            break

    r = img.shape[0]
    for i in range(img.shape[0]):
        r -= 1
        if np.sum(np.asarray(img)[r,:,:]) != 255 * img.shape[1] * img.shape[2]:
            row_bottom = r + margin
            break

    for c in range(img.shape[1]):
        if np.sum(np.asarray(img)[:,c,:]) != 255 * img.shape[0] * img.shape[2]:
            col_left = c - margin
            break

    c = img.shape[1]
    for i in range(img.shape[1]):
        c -= 1
        if np.sum(np.asarray(img)[:,c,:]) != 255 * img.shape[0] * img.shape[2]:
            col_right = c + margin
            break

    # print(row_up, row_bottom, col_left, col_right)
    # pdb.set_trace()

    img = Image.fromarray(img[row_up:row_bottom, col_left:col_right, :])
    img.save(save_path)

    return



def contat_images_anysize_horizontal(imgs):
    """
    https://stackoverflow.com/a/30228789/10509909
    """
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray( i.resize(min_shape) ) for i in imgs))

    imgs_comb = Image.fromarray(imgs_comb)
    return imgs_comb


def contat_images_anysize_vertical(imgs):
    """
    https://stackoverflow.com/a/30228789/10509909
    """
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    imgs_comb = Image.fromarray(imgs_comb)
    return imgs_comb


##############


def concat_images(imga, imgb):
    # Combines two color image ndarrays side-by-side.
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img


def concat_n_images(image_list):
    output = None
    for i, img in enumerate(image_list):
        # img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output


def concat_pic(testcase, config):
    testidx = testcase.split('.png')[0]

    # read data
    pic1 = plt.imread(os.path.join(config['save_path'], testidx, 'X_com.png'))
    pic2 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_com.png'))
    pic3 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_com_hat.png'))
    pic4 = plt.imread(os.path.join(config['save_path'], testidx, 'Colored_Diff.png'))

    interval = np.zeros((pic1.shape[0],  5))
    cm = plt.get_cmap('binary')
    interval = cm(interval)
    interval = (interval[:, :, :3] * 100).astype(np.uint8)

    pics = [pic1, interval, pic2, interval, pic3, interval, pic4]
    output = concat_n_images(pics)


    plt.figure()
    ax = plt.gca()
    im = ax.imshow(output, cmap='jet', interpolation='nearest')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    ax.axis('off')
    # plt.show()
    # os.exit(0)
    plt.savefig(os.path.join(config['save_path'], testidx, 'qualitative.png'), bbox_inches='tight', dpi=1000)

    plt.close('all')
    return


#def concat_pic_compressed1(testcase):
#    testidx = testcase.split('.png')[0]
#
#    # read data
#
#    pic1 = plt.imread(os.path.join(config['save_path'], testidx, 'X_com.png'))
#    pic2 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_com.png'))
#    pic3 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_com_hat.png'))
#    pic4 = plt.imread(os.path.join(config['save_path'], testidx, 'Colored_Diff_com.png'))
#    pic5 = Image.open(os.path.join(config['save_path'], testidx, 'X.png'))
#    pic6 = plt.imread(os.path.join(config['save_path'], testidx, 'Y.png'))
#    pic7 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_hat.png'))
#    pic8 = plt.imread(os.path.join(config['save_path'], testidx, 'Colored_Diff.png'))
#
#    size = 112, 112
#
#    # pic5 = Image.fromarray(pic5)
#    # pic5.thumbnail(size, Image.ANTIALIAS)
#    # pic5 = np.array(pic5)
#
#    pic5 = np.asarray(pic5.resize((112, 112)))
#
#
#    interval = np.zeros((pic1.shape[0],  5))
#    cm = plt.get_cmap('binary')
#    interval = cm(interval)
#    interval = (interval[:, :, :3] * 100).astype(np.uint8)
#
#    pics = [pic1, interval, pic2, interval, pic3, interval, pic4, interval, pic5]
#    #pics = [pic1, interval, pic2, interval, pic3, interval, pic4, interval, pic5, interval, pic6, interval, pic7, interval, pic8]
#    output = concat_n_images(pics)
#
#
#    plt.figure()
#    ax = plt.gca()
#    im = ax.imshow(output, cmap='jet', interpolation='nearest')
#    # create an axes on the right side of ax. The width of cax will be 5%
#    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="1%", pad=0.05)
#    cbar = plt.colorbar(im, cax=cax)
#    cbar.ax.tick_params(labelsize=5)
#    ax.axis('off')
#    # plt.show()
#    # os.exit(0)
#    plt.savefig(os.path.join(config['save_path'], testidx, 'qualitative_all.png'), bbox_inches='tight', dpi=1000)
#
#    return


def concat_pic_compressed(testcase, config):
    testidx = testcase.split('.png')[0]

    # read data

    pic1 = plt.imread(os.path.join(config['save_path'], testidx, 'X_com.png'))
    pic2 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_com.png'))
    pic3 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_com_hat.png'))
    pic4 = plt.imread(os.path.join(config['save_path'], testidx, 'Colored_Diff_com.png'))

    interval = np.zeros((pic1.shape[0],  5))
    cm = plt.get_cmap('binary')
    interval = cm(interval)
    interval = (interval[:, :, :3] * 100).astype(np.uint8)

    pics = [pic1, interval, pic2, interval, pic3, interval, pic4]
    output = concat_n_images(pics)


    plt.figure()
    ax = plt.gca()
    im = ax.imshow(output, cmap='jet', interpolation='nearest')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    ax.axis('off')
    # plt.show()
    # os.exit(0)
    plt.savefig(os.path.join(config['save_path'], testidx, 'qualitative_com.png'), bbox_inches='tight', dpi=1000)



    pic1 = plt.imread(os.path.join(config['save_path'], testidx, 'X.png'))
    pic2 = plt.imread(os.path.join(config['save_path'], testidx, 'Y.png'))
    pic3 = plt.imread(os.path.join(config['save_path'], testidx, 'Y_hat.png'))
    pic4 = plt.imread(os.path.join(config['save_path'], testidx, 'Colored_Diff.png'))

    interval = np.zeros((pic1.shape[0],  5))
    cm = plt.get_cmap('binary')
    interval = cm(interval)
    interval = (interval[:, :, :3] * 100).astype(np.uint8)

    pics = [pic1, interval, pic2, interval, pic3, interval, pic4]
    output = concat_n_images(pics)


    plt.figure()
    ax = plt.gca()
    im = ax.imshow(output, cmap='jet', interpolation='nearest')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    ax.axis('off')
    # plt.show()
    # os.exit(0)
    plt.savefig(os.path.join(config['save_path'], testidx, 'qualitative.png'), bbox_inches='tight', dpi=1000)

    plt.close('all')
    return


if __name__ == "__main__":
    folder = "/Users/zhouhonglu/Downloads/Data_CR_test_pipeline_eval/1001/"
    files = [folder + file for file in os.listdir(folder) if '.png' in file]
    print(os.listdir(folder))
    # pdb.set_trace()

    contat_images_anysize_withpadding_horizontal(files, folder + 'img.png')






