from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
from utils import visualize
import numpy as np
import os
import argparse

def main(args):

    debug = 1

    pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

    image_num = 1

    #predictions
    predfile = args.result
    preds = loadmat(predfile)['preds']
    pos_pred_src = transpose(preds, [1, 2, 0])


    if debug:

        for i in range(image_num):

            imagePath = '/data3/wzwu/dataset/my/' + str(i)+ '.jpg'
            oriImg = sio.imread(imagePath)
            pred = pos_pred_src[:, :, i]
            visualize(oriImg, pred, pa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPII PCKh Evaluation')
    parser.add_argument('-r', '--result', default='../checkpoint/mpii/hg_s2_b1/preds_valid.mat',
                        type=str, metavar='PATH',
                        help='path to result (default: checkpoint/mpii/hg_s2_b1/preds_valid.mat)')
    args = parser.parse_args()
    main(args)
