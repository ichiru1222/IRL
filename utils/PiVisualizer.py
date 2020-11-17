# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:20:10 2017

@author: Toshiaki
"""

import csv

#矢印を作るだけのプログラム
def CompassDirection(Policy):
    if Policy == 0:
        arrow = "↑"
    elif Policy == 1:
        arrow = "↓"
    elif Policy == 2:
        arrow = "←"
    elif Policy == 3:
        arrow = "→"
    elif Policy == 4:
        arrow = "↺"
    return arrow

def CompassDirections(Policies):
    N = len(Policies)
    arrows = []
    
    for i in range(N):
        if Policies[i] != Policies[i]: #nanの判定
            arrows.append("G")
        elif Policies[i] == 0:
            arrows.append("↑")
        elif Policies[i] == 1:
            arrows.append("↓")
        elif Policies[i] == 2:
            arrows.append("←")
        elif Policies[i] == 3:
            arrows.append("→")
        elif Policies[i] == 4:
            arrows.append("↺")

    return arrows

#完全な方策を受け取ってMapに表示する
def GridMapGenerator(arrows,X,Y, save = False, savename="最適方策"):
    field = [["0" for row in range(X)] for column in range(Y)]
    
    for i in range(Y):
        for j in range(X):
            #print(i*X + j)
            field[i][j] = arrows[i*X+j]
    
    for i in range(Y):
        print(field[i])
        
    if save:
        with open(savename + ".csv", "w", encoding="Utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(field)
        
    return field