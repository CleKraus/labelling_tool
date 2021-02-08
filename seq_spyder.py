#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:07:05 2021

@author: makraus
"""

import os

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import seaborn as sns
from skimage import morphology

img_id = '1380'
img_path = os.path.join('boneage-training-dataset', '{}.png'.format(img_id))
img = cv.imread(img_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def windowing(src, pre_cut=True, pieces=5):
    whole_result = np.copy(src)
    if pre_cut:
        while True:
            fig, ax1 = plt.subplots(1, 1)
            ax1.imshow(whole_result)
            plt.show()
            try:
                nb1_y, nb2_y = input('Min Max Height: ').split()
                nb1_x, nb2_x = input('Min Max Width: ').split()
            except:
                continue
            if int(nb1_y) == -1 or int(nb2_y) == -1 or int(nb1_x) == -1 or int(nb2_x) == -1:
                break
            zero = np.zeros(src.shape)
            zero[int(nb1_y): int(nb2_y), int(nb1_x): int(nb2_x)] = whole_result[int(nb1_y): int(nb2_y), int(nb1_x): int(nb2_x)]
            whole_result = zero
            
        src[whole_result == 0.] = 0.
        
    print(whole_result.shape)
    for i in range(pieces):
        for j in range(pieces):
            piece = src[i*int(src.shape[0]/pieces):(i+1)*int(src.shape[0]/pieces),
                        j*int(src.shape[1]/pieces):(j+1)*int(src.shape[1]/pieces)]
            nb = 255
            threshold = 110
            while nb != -1:
                segmentation = piece > threshold
                segmentation = morphology.remove_small_objects(segmentation, 256)
                segmentation = morphology.remove_small_holes(segmentation, 64)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(piece)
                ax2.imshow(segmentation, cmap='gray')
                sns.distplot(piece, ax=ax3)
                plt.show()
                try:
                    nb = int(input('Choose a number: '))
                except:
                    nb = 0
                if nb >= 0:
                    threshold = nb
                elif nb == -10:
                    segmentation = windowing(piece, pre_cut=False, pieces=2)
                    break
                else:
                    break
            whole_result[i*int(src.shape[0]/pieces):(i+1)*int(src.shape[0]/pieces),
                        j*int(src.shape[1]/pieces):(j+1)*int(src.shape[1]/pieces)] = segmentation
    return whole_result

def export_figure_matplotlib(arr, f_name, dpi=200, resize_fact=1, plt_show=False):
    """
    Export array as figure in original resolution
    :param arr: array of image to save in original resolution
    :param f_name: name of file where to save figure
    :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
    :param dpi: dpi of your screen
    :param plt_show: show plot or not
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap='gray')
    plt.savefig(f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()

segmentation = windowing(gray, pieces=3)
segmentation = 255 * (segmentation > 0)
export_figure_matplotlib(segmentation, os.path.join('boneage-training-dataset', '{}_seg.png'.format(img_id)))
