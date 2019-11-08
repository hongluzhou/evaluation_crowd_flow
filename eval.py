#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:50:40 2019

@author: zhouhonglu
"""


import os
import pandas as pd
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
import pdb
import time
from sklearn.metrics import mean_absolute_error

from contat_img import concat_pic_compressed
from contat_img import contat_images_anysize_withpadding_horizontal
from create_config import create_config
config = create_config()


def qualitative_eval(testcase):
    # read data
    A_com = plt.imread(os.path.join(config['data_path'], config['X_com']['A_com'], testcase))
    G_com = plt.imread(os.path.join(config['data_path'], config['X_com']['G_com'], testcase))
    E_com = plt.imread(os.path.join(config['data_path'], config['X_com']['E_com'], testcase))
    Y_com = plt.imread(os.path.join(config['data_path'], config['Y_com'], testcase))
    Y_com_hat = plt.imread(os.path.join(config['data_path'], config['Y_com_hat'], testcase))

    testidx = testcase.split('.png')[0]

    if not os.path.exists(os.path.join(config['save_path'], testidx)):
        os.makedirs(os.path.join(config['save_path'], testidx))

    """
    Composite input: X_com
    """
    # G, E, A
    cm = plt.get_cmap('jet')
    G_com_new = cm(G_com)
    G_com_new = (G_com_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(E_com.shape[0]):
        for j in range(E_com.shape[1]):
            if A_com[i][j] != 0:  # has agent
                G_com_new[i, j, :] = [255, 255, 255]  # white for agents intial location
            if E_com[i][j] == 0:  # non-navigable
                G_com_new[i][j] = [0, 0, 0]  # black for non-navigable cells
    # goal
    (x_idx, y_idx) = np.where(G_com == np.min(G_com))
    G_com_new[x_idx[int(len(x_idx)/2)], y_idx[int(len(y_idx)/2)], :] = [255,0,255]  # Magenta
    # for i in range(len(x_idx)):
    #     G_com_new[x_idx[i], y_idx[i], :] = [255,0,255]  # Magenta

    imsave(os.path.join(config['save_path'], testidx, 'X_com.png'), G_com_new)


    """
    GT output: Y_com
    """
    # density, enviroment, room
    cm = plt.get_cmap('jet')  # Reds
    Y_com_new = cm(Y_com)
    Y_com_new = (Y_com_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(Y_com.shape[0]):
        for j in range(Y_com.shape[1]):
            if Y_com[i][j] == 0:  # black which is no density
                Y_com_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E_com[i][j] == 0:  # non-navigable
                Y_com_new[i, j, :] = 0  # black
            # if A_com[i][j] != 0:  # has agent
            #     Y_com_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Y_com.png'), Y_com_new)

    """
    Predict output: Y_com_hat
    """
    # density, enviroment, room
    cm = plt.get_cmap('jet')  # Reds
    Y_com_hat_new = cm(Y_com_hat)
    Y_com_hat_new = (Y_com_hat_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(Y_com_hat.shape[0]):
        for j in range(Y_com_hat.shape[1]):
            if Y_com_hat[i][j] == 0:  # black which is no density
                Y_com_hat_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E_com[i][j] == 0:  # non-navigable
                Y_com_hat_new[i, j, :] = 0  # black
            # if A_com[i][j] != 0:  # has agent
            #     Y_com_hat_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Y_com_hat.png'), Y_com_hat_new)


    """
    Colored difference
    """
    Diff = np.zeros(Y_com.shape)
    for i in range(Y_com.shape[0]):
        for j in range(Y_com.shape[1]):
            Diff[i][j] = Y_com_hat[i][j] - Y_com[i][j]
            Diff[i][j] = Diff[i][j]/2 + 0.5
    # print(Diff)

    cm = plt.get_cmap('jet')  # Reds
    Diff_new = cm(Diff)
    Diff_new = (Diff_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(E_com.shape[0]):
        for j in range(E_com.shape[1]):
            if ((Y_com[i][j] == 0) and (Y_com_hat[i][j] == 0)):  # black, no one walks on it
                Diff_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E_com[i][j] == 0:  # non-navigable
                Diff_new[i, j, :] = 0  # black
            # if A_com[i][j] != 0:  # has agent
            #     Diff_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Colored_Diff_com.png'), Diff_new)


    """
    concaticate all qualitative results of this test case
    """
    #concat_pic(testcase)
    files = ['X_com.png', 'Y_com.png', 'Y_com_hat.png', 'Colored_Diff_com.png']
    files_fullpath = [os.path.join(config['save_path'], testidx, file) for file in files]
    contat_images_anysize_withpadding_horizontal(files_fullpath, os.path.join(config['save_path'], testidx, 'qualitative.png'))


    return


def qualitative_eval_compressed(testcase):
    # read data
    A = plt.imread(os.path.join(config['data_path'], config['X']['A'], testcase))
    G = plt.imread(os.path.join(config['data_path'], config['X']['G'], testcase))
    E = plt.imread(os.path.join(config['data_path'], config['X']['E'], testcase))
    A_com = plt.imread(os.path.join(config['data_path'], config['X_com']['A_com'], testcase))
    G_com = plt.imread(os.path.join(config['data_path'], config['X_com']['G_com'], testcase))
    E_com = plt.imread(os.path.join(config['data_path'], config['X_com']['E_com'], testcase))
    Y_com = plt.imread(os.path.join(config['data_path'], config['Y_com'], testcase))
    Y_com_hat = plt.imread(os.path.join(config['data_path'], config['Y_com_hat'], testcase))
    Y = plt.imread(os.path.join(config['data_path'], config['Y'], testcase))
    Y_hat = plt.imread(os.path.join(config['data_path'], config['Y_hat'], testcase))

    testidx = testcase.split('.png')[0]

    if not os.path.exists(os.path.join(config['save_path'], testidx)):
        os.makedirs(os.path.join(config['save_path'], testidx))


    """
    Raw composite input: X
    """
    # G, E, A
    cm = plt.get_cmap('jet')
    G_new = cm(G)
    G_new = (G_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if A[i][j] == 1:  # has agent
                G_new[i, j, :] = [255, 255, 255]  # white for agents intial location
            if E[i][j] == 0:  # non-navigable
                G_new[i][j] = [0, 0, 0]  # black for non-navigable cells

    # goal
    (x_idx, y_idx) = np.where(G == np.min(G))
    G_new[x_idx[int(len(x_idx)/2)], y_idx[int(len(y_idx)/2)], :] = [255,0,255]  # Magenta
    # for i in range(len(x_idx)):
    #     G_com_new[x_idx[i], y_idx[i], :] = [255,0,255]  # Magenta

    imsave(os.path.join(config['save_path'], testidx, 'X.png'), G_new)


    """
    Compressed composite input: X_com
    """
    # G, E, A
#    A_com_tem = A_com.copy()
#    for i in range(A_com.shape[0]):
#        for j in range(A_com.shape[1]):
#            if A_com[i][j] > 0 and A_com[i][j] < 1:  # has agent
#                A_com_tem[i][j] = rgb_grey_color(A_com[i][j])
#                pdb.set_trace()

    gm = plt.get_cmap('gray')
    A_com_tem = (gm(A_com)[:, :, :3] * 255).astype(np.uint8)

    cm = plt.get_cmap('jet')
    G_com_new = cm(G_com)
    G_com_new = (G_com_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(E_com.shape[0]):
        for j in range(E_com.shape[1]):
            if A_com[i][j] != 0:  # has agent
                if A_com[i][j] == 1:
                    G_com_new[i, j, :] = [255, 255, 255]  # white for agents intial location
                if A_com[i][j] < 1:
                    G_com_new[i, j, :] = A_com_tem[i, j, :]
            if E_com[i][j] == 0:  # non-navigable
                G_com_new[i][j] = [0, 0, 0]  # black for non-navigable cells


    # goal
    (x_idx, y_idx) = np.where(G_com == np.min(G_com))
    G_com_new[x_idx[int(len(x_idx)/2)], y_idx[int(len(y_idx)/2)], :] = [255,0,255]  # Magenta
    # for i in range(len(x_idx)):
    #     G_com_new[x_idx[i], y_idx[i], :] = [255,0,255]  # Magenta

    imsave(os.path.join(config['save_path'], testidx, 'X_com.png'), G_com_new)


    """
    Compressed GT output: Y_com
    """
    # density, enviroment, room
    cm = plt.get_cmap('jet')  # Reds
    Y_com_new = cm(Y_com)
    Y_com_new = (Y_com_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(Y_com.shape[0]):
        for j in range(Y_com.shape[1]):
            if Y_com[i][j] == 0:  # black which is no density
                Y_com_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E_com[i][j] == 0:  # non-navigable
                Y_com_new[i, j, :] = 0  # black
            # if A_com[i][j] != 0:  # has agent
            #     Y_com_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Y_com.png'), Y_com_new)

    """
    Compressed predict output: Y_com_hat
    """
    # density, enviroment, room
    cm = plt.get_cmap('jet')  # Reds
    Y_com_hat_new = cm(Y_com_hat)
    Y_com_hat_new = (Y_com_hat_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(Y_com_hat.shape[0]):
        for j in range(Y_com_hat.shape[1]):
            if Y_com_hat[i][j] == 0:  # black which is no density
                Y_com_hat_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E_com[i][j] == 0:  # non-navigable
                Y_com_hat_new[i, j, :] = 0  # black
            # if A_com[i][j] != 1:  # has agent
            #     Y_com_hat_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Y_com_hat.png'), Y_com_hat_new)


    """
    Raw GT output: Y
    """
    # density, enviroment, room
    cm = plt.get_cmap('jet')  # Reds
    Y_new = cm(Y)
    Y_new = (Y_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i][j] == 0:  # black which is no density
                Y_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E[i][j] == 0:  # non-navigable
                Y_new[i, j, :] = 0  # black
            # if A[i][j] == 1:  # has agent
            #     Y_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Y.png'), Y_new)

    """
    Raw predict output: Y_hat
    """
    # density, enviroment, room
    cm = plt.get_cmap('jet')  # Reds
    Y_hat_new = cm(Y_hat)
    Y_hat_new = (Y_hat_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(Y_hat.shape[0]):
        for j in range(Y_hat.shape[1]):
            if Y_hat[i][j] == 0:  # black which is no density
                Y_hat_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E[i][j] == 0:  # non-navigable
                Y_hat_new[i, j, :] = 0  # black
            # if A[i][j] == 1:  # has agent
            #     Y_hat_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Y_hat.png'), Y_hat_new)


    """
    Compressed colored difference
    """
    Diff = np.zeros(Y_com.shape)
    for i in range(Y_com.shape[0]):
        for j in range(Y_com.shape[1]):
            Diff[i][j] = Y_com_hat[i][j] - Y_com[i][j]
            Diff[i][j] = Diff[i][j]/2 + 0.5
    # print(Diff)

    cm = plt.get_cmap('jet')  # Reds
    Diff_new = cm(Diff)
    Diff_new = (Diff_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(E_com.shape[0]):
        for j in range(E_com.shape[1]):
            if ((Y_com[i][j] == 0) and (Y_com_hat[i][j] == 0)):  # black, no one walks on it
                Diff_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E_com[i][j] == 0:  # non-navigable
                Diff_new[i, j, :] = 0  # black
            # if A_com[i][j] != 0:  # has agent
            #     Diff_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Colored_Diff_com.png'), Diff_new)


    """
    Colored difference
    """
    Diff = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Diff[i][j] = Y_hat[i][j] - Y[i][j]
            Diff[i][j] = Diff[i][j]/2 + 0.5
    # print(Diff)

    cm = plt.get_cmap('jet')  # Reds
    Diff_new = cm(Diff)
    Diff_new = (Diff_new[:, :, :3] * 255).astype(np.uint8)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if ((Y[i][j] == 0) and (Y_hat[i][j] == 0)):  # black, no one walks on it
                Diff_new[i,j,:] = [192,192,192]  # grey [192,192,192]
            if E[i][j] == 0:  # non-navigable
                Diff_new[i, j, :] = 0  # black
            # if A[i][j] == 1:  # has agent
            #     Diff_new[i, j, :] = [255, 255, 0]  # yellow
    imsave(os.path.join(config['save_path'], testidx, 'Colored_Diff.png'), Diff_new)



    """
    concaticate all qualitative results of this test case
    """
    G = plt.imread(os.path.join(config['data_path'], config['X']['G'], testcase))
    # goal
    (x_idx, y_idx) = np.where(G == np.min(G))
    G_idx = [int(x_idx[int(len(x_idx)/2)]/2), int(y_idx[int(len(y_idx)/2)]/2)]
    # print(G_idx)
    # pdb.set_trace()
    # G_new[x_idx[int(len(x_idx)/2)], y_idx[int(len(y_idx)/2)], :] = [255,0,255]  # Magenta



    files = ['X.png', 'X_com.png', 'Y_com.png', 'Y_com_hat.png', 'Colored_Diff_com.png', 'Y.png', 'Y_hat.png', 'Colored_Diff.png']
    files_fullpath = [os.path.join(config['save_path'], testidx, file) for file in files]
    contat_images_anysize_withpadding_horizontal(files_fullpath, os.path.join(config['save_path'], testidx, 'qualitative.png'), G_idx)
    # concat_pic_compressed(testcase, config)

    return


def quantitative_eval(testcase):
    Y_com = plt.imread(os.path.join(config['data_path'], config['Y_com'], testcase))
    Y_com_hat = plt.imread(os.path.join(config['data_path'], config['Y_com_hat'], testcase))

    mae = mean_absolute_error(Y_com, Y_com_hat)
    kl = KLDivergence(Y_com, Y_com_hat)

    return  [None, None, kl, mae]


def quantitative_eval_compressed(testcase):
    Y_com = plt.imread(os.path.join(config['data_path'], config['Y_com'], testcase))
    Y_com_hat = plt.imread(os.path.join(config['data_path'], config['Y_com_hat'], testcase))
    Y = plt.imread(os.path.join(config['data_path'], config['Y'], testcase))
    Y_hat = plt.imread(os.path.join(config['data_path'], config['Y_hat'], testcase))

    mae_com = mean_absolute_error(Y_com, Y_com_hat)
    kl_com = KLDivergence(Y_com, Y_com_hat)

    mae = mean_absolute_error(Y, Y_hat)
    kl = KLDivergence(Y, Y_hat)

    return  [kl_com, mae_com, kl, mae]


    return


#def KLDivergence(a, b):
#    """
#    https://datascience.stackexchange.com/a/9264
#    """
#    a = np.asarray(a, dtype=np.float)
#    b = np.asarray(b, dtype=np.float)
#
#    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def KLDivergence(p, q):
    # p is ground truth, q is prediction
    p = p.reshape((np.square(p.shape[0]),))
    q = q.reshape((np.square(q.shape[0]),))

    p = p/np.sum(p)
    q = q/np.sum(q)


    result = 0
    for i in range(len(p)):
        if q[i] != 0 and p[i] !=0:
            tem = p[i] * np.log(p[i]/q[i])
            result += tem
    return result



if __name__ == "__main__":
    # get all test cases
    all_testcases = [file for file in os.listdir(os.path.join(config['data_path'], config['X_com']['A_com'])) if '.png' in file]
    all_testcases = ['1.png']


    if not os.path.exists(os.path.join(config['save_path'])):
        os.makedirs(os.path.join(config['save_path']))


    if config['quantitative']:
        MAE_compressed = []
        KL_compressed = []
        MAE = []
        KL = []

    if config['compression_rate'] == 1:
        for i in range(len(all_testcases)):
            testcase = all_testcases[i]
            # start_time = time.time()

            if config['quantitative']:
                [kl_com, mae_com, kl, mae] = quantitative_eval(testcase)
                KL_compressed.append(kl_com)
                MAE_compressed.append(mae_com)
                KL.append(kl)
                MAE.append(mae)

            if config['qualitative']:
                qualitative_eval(testcase)

            # print("{} took: {}".format(testcase, time.time() - start_time))
    else:
        for i in range(len(all_testcases)):
            testcase = all_testcases[i]
            # start_time = time.time()

            if config['quantitative']:
                [kl_com, mae_com, kl, mae] = quantitative_eval_compressed(testcase)
                KL_compressed.append(kl_com)
                MAE_compressed.append(mae_com)
                KL.append(kl)
                MAE.append(mae)

            if config['qualitative']:
                qualitative_eval_compressed(testcase)

            # print("{} took: {}".format(testcase, time.time() - start_time))

    if config['quantitative']:
        data = {'testcase': [testcase.split('.png')[0] for testcase in all_testcases]}

        data['MAE_decompressed'] = MAE_compressed
        data['KL_decompressed'] = KL_compressed
        data['MAE'] = MAE
        data['KL'] = KL
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(config['save_path'], 'quantitative.csv'), index=False)


        print("mean MAE: {}".format(np.mean(MAE)))
        print("std MAE: {}".format(np.std(MAE)))
        print("mean KL: {}".format(np.mean(KL)))
        print("std KL: {}".format(np.std(KL)))

        if config['compression_rate'] > 1:
            print("mean MAE_decompressed: {}".format(np.mean(MAE_compressed)))
            print("std MAE_decompressed: {}".format(np.std(MAE_compressed)))
            print("mean KL_decompressed: {}".format(np.mean(KL_compressed)))
            print("std KL_decompressed: {}".format(np.std(KL_compressed)))

        # save overall metrics
        overall = dict()
        overall['mean MAE'] = [np.mean(MAE)]
        overall['std MAE'] = [np.std(MAE)]
        overall['mean KL'] = [np.mean(KL)]
        overall['std KL'] = [np.std(KL)]
        overall['mean MAE_decompressed'] = [np.mean(MAE_compressed)]
        overall['std MAE_decompressed'] = [np.std(MAE_compressed)]
        overall['mean KL_decompressed'] = [np.mean(KL_compressed)]
        overall['std KL_decompressed'] = [np.std(KL_compressed)]
        overall['num testcases'] = [len(all_testcases)]
        df = pd.DataFrame(overall)
        df.to_csv(os.path.join(config['save_path'], 'quantitative_overall.csv'), index=False)

