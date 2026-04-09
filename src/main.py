# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--cuda', '-c', type=str, default='0', help='cuda device id')

    parser.add_argument('--mask_weight_f', type=float, default=1e-1, help='cuda device id')
    parser.add_argument('--mask_weight_g', type=float, default=1e-1, help='cuda device id')
    parser.add_argument('--align_weight', type=float, default=1e-1, help='cuda device id')
    parser.add_argument('--missing_rate', type=float, default=1e-1, help='cuda device id')


    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': int(args.cuda),
        'mask_weight_f': args.mask_weight_f,
        'mask_weight_g': args.mask_weight_g,
        'align_weight': args.align_weight,
        'missing_rate':args.missing_rate
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
