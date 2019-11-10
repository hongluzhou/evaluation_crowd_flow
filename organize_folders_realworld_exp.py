#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:10:16 2019

@author: zhouhonglu
"""

import os
import pdb
import shutil


def create_config():
    config = {
            'morning':
                {'10': ['A', '10'],
                 '11':	['C', '10'],
                 '12':	['A', '6'],
                 '13':	['B'	, '10'],
                 '15':	['A'	, '10'],
                 '16':	['B'	, '10'],
                 '17':	['B'	, '10'],
                 '18':	['A'	, '10'],
                 '19':	['B'	, '10'],
                 '35':	['A'	, '7'],
                 '36':	['A'	, '7'],
                 '37':	['A'	, '7'],
                 '38':	['A'	, '6'],
                 '39':	['A'	, '6'],
                 '40':	['B'	, '10'],
                 '41':	['B'	, '10'],
                 '42':	['A'	, '1'],
                 '43':	['C'	, '10'],
                 '44':	['A'	, '10'],
                 '45':	['B'	, '10']
                 },

            'afternoon':
                {'3': ['A', '9'],
                 '4': ['A', '10'],
                 '5': ['A', '1'],
                 '6': ['A', '1'],
                 '7': ['A', '1'],
                 '8': ['A', '1'],
                 '9': ['A', '1'],
                 '10': ['A', '6'],
                 '11': ['A', '6'],
                 '12': ['A', '10'],
                 '13': ['A', '10'],
                 '14': ['A', '6'],
                 '15': ['A', '6'],
                 '16': ['B', '10'],
                 '17': ['A', '10'],
                 '18': ['A', '10'],
                 '19': ['A', '10'],
                 '20': ['A', '1'],
                 '21': ['A', '1'],
                 '22': ['A', '1'],
                 '23': ['A', '6'],
                 '24': ['A', '7'],
                 '25': ['A', '7'],
                 '26': ['A', '7'],
                 '27': ['A', '10'],
                 '28': ['A', '7'],
                 '29': ['A', '7'],
                 '30': ['A', '1'],
                 '31': ['A', '8'],
                 '32': ['A', '1'],
                 '33': ['A', '6'],
                 '34': ['A', '10'],
                 '35': ['A', '1'],
                 '36': ['A', '6'],
                 '37': ['A', '6'],
                 '38': ['A', '1'],
                 '39': ['A', '6'],
                 '40': ['A', '6'],
                 '41': ['A', '8'],
                 '42': ['A', '10'],
                 '43': ['B', '10'],
                 '44': ['A', '10'],
                 '45': ['A', '1']
                 },

            'model_name': 'M8',

            'data_path': '/Users/zhouhonglu/Downloads/M8_test_on_real_data',

            'save_path': '/Users/zhouhonglu/Downloads/M8onRW',

            'compression_rate': 1,

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
            'Y_com_hat': 'Prediction_Comp',

            # Raw GT output (when compression_rate > 1)
            'Y': 'Proxy_Decomp',

            # Raw Pred output (when compression_rate > 1)
            'Y_hat': 'decompP'
    }
    return config


if __name__ == '__main__':
    config = create_config()

    unique_los = set()
    unique_directions = set()
    for tc in config['morning']:
        unique_los.add(config['morning'][tc][0])
        unique_directions.add(config['morning'][tc][1])
    for tc in config['afternoon']:
        unique_los.add(config['afternoon'][tc][0])
        unique_directions.add(config['afternoon'][tc][1])

    print(unique_los, len(unique_los))
    print(unique_directions, len(unique_directions))

    for los in unique_los:
        if not os.path.exists(os.path.join(config['save_path'], config['model_name'] + 'onLOS' + los)):
            os.makedirs(os.path.join(config['save_path'], config['model_name'] + 'onLOS' + los))

    for dr in unique_directions:
        if not os.path.exists(os.path.join(config['save_path'], config['model_name'] + 'onDirection' + dr)):
            os.makedirs(os.path.join(config['save_path'], config['model_name'] + 'onDirection' + dr))

    if config['compression_rate'] == 1:
        for mor_aft in ['afternoon', 'morning']:
            for tc in config[mor_aft]:
                # LoS
                old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['A_com'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onLOS' + config[mor_aft][tc][0], config['X_com']['A_com'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['E_com'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onLOS' + config[mor_aft][tc][0], config['X_com']['E_com'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                if mor_aft == 'afternoon':
                    old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['G_com'])
                    new_path = os.path.join(config['save_path'], config['model_name'] + 'onLOS' + config[mor_aft][tc][0], config['X_com']['G_com'])
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copyfile(
                            os.path.join(old_path, str(int(tc) - 3) + '.png'),
                            os.path.join(new_path, tc + '_' + mor_aft + '.png')
                            )
                else:
                    old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['G_com'])
                    new_path = os.path.join(config['save_path'], config['model_name'] + 'onLOS' + config[mor_aft][tc][0], config['X_com']['G_com'])
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copyfile(
                            os.path.join(old_path, tc + '.png'),
                            os.path.join(new_path, tc + '_' + mor_aft + '.png')
                            )

                old_path = os.path.join(config['data_path'], mor_aft, config['Y_com'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onLOS' + config[mor_aft][tc][0], config['Y_com'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                old_path = os.path.join(config['data_path'], mor_aft, config['Y_com_hat'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onLOS' + config[mor_aft][tc][0], config['Y_com_hat'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                # Direction
                old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['A_com'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onDirection' + config[mor_aft][tc][1], config['X_com']['A_com'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['E_com'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onDirection' + config[mor_aft][tc][1], config['X_com']['E_com'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                if mor_aft == 'afternoon':
                    old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['G_com'])
                    new_path = os.path.join(config['save_path'], config['model_name'] + 'onDirection' + config[mor_aft][tc][1], config['X_com']['G_com'])
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copyfile(
                            os.path.join(old_path, str(int(tc) - 3) + '.png'),
                            os.path.join(new_path, tc + '_' + mor_aft + '.png')
                            )
                else:
                    old_path = os.path.join(config['data_path'], mor_aft, config['X_com']['G_com'])
                    new_path = os.path.join(config['save_path'], config['model_name'] + 'onDirection' + config[mor_aft][tc][1], config['X_com']['G_com'])
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copyfile(
                            os.path.join(old_path, tc + '.png'),
                            os.path.join(new_path, tc + '_' + mor_aft + '.png')
                            )

                old_path = os.path.join(config['data_path'], mor_aft, config['Y_com'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onDirection' + config[mor_aft][tc][1], config['Y_com'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )

                old_path = os.path.join(config['data_path'], mor_aft, config['Y_com_hat'])
                new_path = os.path.join(config['save_path'], config['model_name'] + 'onDirection' + config[mor_aft][tc][1], config['Y_com_hat'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                shutil.copyfile(
                        os.path.join(old_path, tc + '.png'),
                        os.path.join(new_path, tc + '_' + mor_aft + '.png')
                        )
    else:
        print("Not implemented for compression experiment!")
        os._exit(0)

