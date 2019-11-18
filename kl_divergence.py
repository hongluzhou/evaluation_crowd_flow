#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:48:16 2019

@author: zhouhonglu
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import pdb


def kl_divergence(p, q):
    """
    my implementation
    """
    result = 0
    for i in range(len(p)):
        tem = p[i] * np.log(p[i]/q[i])
        result += tem
    return result


def KL(a, b):
    """
    https://datascience.stackexchange.com/a/9264
    """
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def mse(p, q):
    """
    my implementation
    """
    result = 0
    for i in range(len(p)):
        tem = np.square(p[i] - q[i])
        result += tem
    return result/len(p)


q = [0.36438797, 0.12192962, 0.19189483, 0.32178759]
p_1 = [0.37300551, 0.12188121, 0.18509246, 0.32002081]
p_2 = [0.33014522, 0.03053611, 0.30264458, 0.33667409]

a = round(kl_divergence(p_1, q), 10)
b = round(mse(p_1, q), 10)
c = round(kl_divergence(p_2, q), 10)
d = round(mse(p_2, q), 10)
print(a)
print(b)
print(c)
print(d)
print(a>c)
print(b>d)