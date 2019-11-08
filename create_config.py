#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:51:52 2019

@author: zhouhonglu
"""


def create_config():
    config = {

            'data_path': '/Users/zhouhonglu/Downloads/M8onCompressionExp',
            # where you have inputs, ground truth, and predictions

            'save_path': '/Users/zhouhonglu/Downloads/M8onCompressionExp_eval',
            # where you save evaluation results

            'qualitative': True,
            'quantitative': False,

            'compression_rate': 2,

            # Raw input (when compression_rate > 1)
            'X': {
                    'A': 'A_Original',
                    'G': 'G_Decomp',
                    'E': 'E_Decomp'
                    },

            # Compressed input / input to model
            'X_com': {
                    'A_com': 'A_Comp',
                    'G_com': 'G_Comp',
                    'E_com': 'E_Comp'
                    },

            # Compressed GT output / GT output from model
            'Y_com': 'Proxy_Comp',

            # Compressed Pred output / Pred output from model
            'Y_com_hat': 'prediction',

            # Raw GT output (when compression_rate > 1)
            'Y': 'Proxy_Decomp',

            # Raw Pred output (when compression_rate > 1)
            'Y_hat': 'decompP'
    }
    return config
