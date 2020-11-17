# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:41:49 2019

@author: Toshiaki
"""
import numpy as np
np.set_printoptions(threshold=np.inf)

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

import copy
from collections import defaultdict

def scored_sort_rewardnet(traj, model):

    traj_list = []
    for i in range(len(traj)):
        traj_list.append(i)
    
    sumR = np.zeros(len(traj))
    traj_sumR_dict = defaultdict(float)
    
    for i, key in enumerate(traj_list):
        tmp_traj = Variable(traj[i].astype(np.float32).reshape(len(traj[i]),1))
        sumR[i] = F.sum(model(tmp_traj)).data
        traj_sumR_dict[key] += sumR[i]
    
    #print("ランキング前")    
    #print(traj_sumR_dict)
    
    ranking = sorted(traj_sumR_dict.items(), key=lambda x: -x[1])
    #print("ランキング後")
    #print(ranking)
    
    ranked_traj = copy.deepcopy(traj) 
    
    for i in range(len(traj)):
        ranked_traj[i] = traj[ranking[i][0]]
    
    return ranked_traj

def ave_ranked_sort(traj, ave_rank):

    traj_list = []
    for i in range(len(traj)):
        traj_list.append(i)
    
    traj_sumR_dict = defaultdict(float)
    
    for i, key in enumerate(traj_list):
        traj_sumR_dict[key] += ave_rank[i]
    
    #print("ランキング前")    
    #print(traj_sumR_dict)
    
    ranking = sorted(traj_sumR_dict.items(), key=lambda x: x[1])
    #print("ランキング後")
    #print(ranking)
    
    ranked_traj = copy.deepcopy(traj) 
    
    for i in range(len(traj)):
        ranked_traj[i] = traj[ranking[i][0]]
    
    return ranked_traj

def scored_sort(traj, R):

    traj_list = []
    for i in range(len(traj)):
        traj_list.append(i)
    
    sumR = np.zeros(len(traj))
    traj_sumR_dict = defaultdict(float)
    
    for i, key in enumerate(traj_list):
        for s in traj[i]:
            sumR[i] += R[s] #初期状態のものも入ってしまうが，初期状態が同じなら特に差が生まれないのでよしとしている
        traj_sumR_dict[key] += sumR[i]
    
    #print("ランキング前")    
    #print(traj_sumR_dict)
    
    ranking = sorted(traj_sumR_dict.items(), key=lambda x: -x[1])
    #print("ランキング後")
    #print(ranking)
    
    ranked_traj = copy.deepcopy(traj) 
    
    for i in range(len(traj)):
        ranked_traj[i] = traj[ranking[i][0]]
    
    return ranked_traj

def discount_scored_sort(traj, R, gamma):

    traj_list = []
    for i in range(len(traj)):
        traj_list.append(i)
    
    sumR = np.zeros(len(traj))
    traj_sumR_dict = defaultdict(float)
    
    for i, key in enumerate(traj_list):
        for t, s in enumerate(traj[i]):
            #初期状態のものも入ってしまうが，初期状態が同じなら特に差が生まれないのでよしとしている
            sumR[i] += (gamma**t)*R[s]
        traj_sumR_dict[key] += sumR[i]
    
    #print("ランキング前")    
    #print(traj_sumR_dict)
    
    ranking = sorted(traj_sumR_dict.items(), key=lambda x: -x[1])
    #print("ランキング後")
    #print(ranking)
    
    ranked_traj = copy.deepcopy(traj) 
    
    for i in range(len(traj)):
        ranked_traj[i] = traj[ranking[i][0]]
    
    return ranked_traj

def reliability_sort(traj, reliability):
    #trajの順番とreliabilityの順番は一致していることを仮定
    
    traj_list = []
    for i in range(len(traj)):
        traj_list.append(i)
    
    traj_dict = defaultdict(float)
    
    for i, key in enumerate(traj_list):
        traj_dict[key] += reliability[i]
    
    #print("ランキング前")  
    #print(traj_dict)
    
    ranking = sorted(traj_dict.items(), key=lambda x: -x[1])
    #print("ランキング後")
    #print(ranking)
    
    ranked_traj = copy.deepcopy(traj) 
    
    for i in range(len(traj)):
        ranked_traj[i] = traj[ranking[i][0]]
    
    return ranked_traj
