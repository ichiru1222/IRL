# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:37:14 2018

@author: Toshiaki
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def GridHeatMap(reward, Xsize, Ysize, save = False, savename = None,  reverse = True):
    #Reward配列は[R(S1),R(S2),...,R(Sn)]の順に報酬が格納されている配列  
    
    list_2D = []
    
    for i in range(Ysize):
        list_2D.append(reward[i*Xsize:(i+1)*Xsize])
    
    plt.figure()
    sns.set(font_scale=1.5)
    ax = sns.heatmap(list_2D, cmap="RdBu_r")
    ax.invert_yaxis()
    
    if save:
        plt.savefig(savename + ".png")
        plt.close('all')
    