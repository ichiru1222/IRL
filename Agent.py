# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:42:07 2018

@author: D. Kishikawa
"""
import numpy as np

class Agent():
    def __init__(self, Sekai, Goal):
     
        self.X = Sekai.X_size
        self.Y = Sekai.Y_size
        
        self.nowPos = 0
        self.nextPos = 0
        
        self.goal = Goal
        
        self.R = 0
        
        #Qtableの用意
        self.Q = np.zeros((self.X * self.Y, 4))