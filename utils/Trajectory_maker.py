# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:20:10 2017

@author: Toshiaki
"""

import numpy as np
import math
#報酬R(状態数),状態遷移確率P(行動数*状態数*状態数),状態S(状態数),行動A(行動数)
#それぞれ（）内の数の要素を持った配列
np.random.seed(0)

def GlidTransition(X,Y,nowstate,Goal,act):
    #画面外に出る場合元の状態に戻る
    N = X*Y
    if act == 0:    #上
        if nowstate in Goal:
            afterstate = nowstate
        elif nowstate >= N-X:
            afterstate = nowstate
        else:
            afterstate = nowstate + X

    elif act == 1:    #下
        if nowstate in Goal:
            afterstate = nowstate
        elif nowstate < X:
            afterstate = nowstate
        else:
            afterstate = nowstate - X
        
    elif act == 2:    #左
        if nowstate in Goal:
            afterstate = nowstate
        elif nowstate % X == 0:
            afterstate = nowstate
        else:
            afterstate = nowstate - 1
        
    elif act == 3:    #右
        if nowstate in Goal:
            afterstate = nowstate
        elif (nowstate+1) % X == 0:
            afterstate = nowstate
        else:
            afterstate = nowstate + 1        
        
    return afterstate

def getMaxAction(Qvalue,state,A):
    maxQ = -100
    index = []
    for i in range(len(A)):
        q = Qvalue[state][i] #Q(s,a)
        #最大Qとなる行動を記憶
        if q > maxQ:
            index = [i]
            maxQ = q
        elif q == maxQ:
            index.append(i)
        #print(maxQ)
    return index[np.random.randint(0,len(index))] #インデックスの長さ
    #Q値が最大のものが複数あれば，ランダムで一つ選んで返す

#行動決定
def choose_a(Qvalue,state,A,epsilon):
    if epsilon > np.random.rand(): #0~1の乱数
        a = np.random.randint(0,len(A)) #0 <= N <= A-1であるようなランダムな整数 N を返します
    else:
        #最大化するactionを取得
        a = getMaxAction(Qvalue,state,A)
    return a

def choose_boltzman_a(boltzman_policy, state, A):
    act = np.random.choice(A, p = boltzman_policy[state])
    return act

def culc_boltzman_policy(Q, inv_tempretures):
    boltzman_policy = np.zeros([len(Q), len(Q[0])])
    for s in range(len(Q)):
        sum_exp = 0
        for a in range(len(Q[s])):
            sum_exp += math.exp(inv_tempretures[s] * Q[s][a])
        for a in range(len(Q[s])):
            boltzman_policy[s][a] = math.exp(inv_tempretures[s] * Q[s][a])/sum_exp
    
    return boltzman_policy

#Start:初期状態,S:状態集合,A:行動集合,R:報酬関数
def QLearning_make_trajectory(Start, X, Y, Goal, A, R, epsilon, Episode, min_step = -float('inf'), max_step = float('inf'), truncation = 30, greedy = True):
    ep = 0 
    N = X*Y
    pi = np.zeros(N)
    Qvalue = np.zeros([N, len(A)]) #Q値配列
    alpha = 0.1
    gamma = 0.95
    #delimiter = int(Episode / 10)
    
    while ep < Episode:
        step  = 0
        state = Start
        while True:
            #stepの開始
            step += 1
            #行動の決定(epsilon-greedy)
            a = choose_a(Qvalue,state,A,epsilon)
            #行動後の状態
            afterstate = GlidTransition(X,Y,state,Goal,a) #afterstate
            #次状態で最大の行動・Q値
            maxa = getMaxAction(Qvalue,afterstate,A)
            afterQ = Qvalue[afterstate][maxa]
            #報酬決定
            r = R[afterstate]
            #Q値の更新
            tmp = r + gamma*afterQ - Qvalue[state][a]
            Qvalue[state][a] = Qvalue[state][a] + alpha*tmp
            #print(ep)
    
            if afterstate in Goal or step > truncation:
                ep += 1
                #prog = 100*ep/Episode
                #進捗確認
                #print('エピソード{0}でのstep数:{1}'.format(ep,step))
                #if (ep % delimiter == 0):
                #    print('進捗率{0}％(エピソード{1}まで学習完了)'.format(prog,ep))
                #    print('エピソード{0}でのstep数:{1}'.format(ep,step))
                break
            state = afterstate
        
    for state in range(N):
        pi[state] = np.argmax(Qvalue[state])   

    #軌跡の生成
    s_list = [Start]
    s_a_list = []
    #and (min_step <= len(s_list) <= max_step) 
    while not ( (s_list[-1] in Goal)):
        s_list = [Start]
        s_a_list = []
        state = Start
        
        while not (state in Goal):
            action = choose_a(Qvalue, state, A, epsilon=0)
            s_a_list.append((state, action))
            #行動後の状態
            afterstate = GlidTransition(X,Y,state,Goal,action) #afterstate
            s_list.append(afterstate)
            state = afterstate
            
    dummy = 0
    s_a_list.append((state, dummy))
    #state_trajectory = np.array(s_list, dtype=int)
    #print("軌跡{}，ゴール{}".format(state_trajectory, Goal))
    
    return [Qvalue, pi, s_list, s_a_list]

#Start:初期状態,S:状態集合,A:行動集合,R:報酬関数
def QLearning_make_trajectory_scored(Start, X, Y, Goal, A, R, epsilon, Episode,
                                     min_score, max_score, truncation, greedy = True):
    ep = 0 
    N = X*Y
    pi = np.zeros(N)
    Qvalue = np.zeros([N, len(A)]) #Q値配列
    alpha = 0.2
    gamma = 0.9
    #delimiter = int(Episode / 10)
    
    while ep < Episode:
        step  = 0
        state = Start
        while True:
            #stepの開始
            step += 1
            #行動の決定(epsilon-greedy)
            a = choose_a(Qvalue,state,A,epsilon)
            #行動後の状態
            afterstate = GlidTransition(X,Y,state,Goal,a) #afterstate
            #次状態で最大の行動・Q値
            maxa = getMaxAction(Qvalue,afterstate,A)
            afterQ = Qvalue[afterstate][maxa]
            #報酬決定
            r = R[afterstate]
            #Q値の更新
            tmp = r + gamma*afterQ - Qvalue[state][a]
            Qvalue[state][a] = Qvalue[state][a] + alpha*tmp
            #print(ep)
    
            if afterstate in Goal or step > truncation:
                ep += 1
                #prog = 100*ep/Episode
                #進捗確認
                #print('エピソード{0}でのstep数:{1}'.format(ep,step))
                #if (ep % delimiter == 0):
                #    print('進捗率{0}％(エピソード{1}まで学習完了)'.format(prog,ep))
                #    print('エピソード{0}でのstep数:{1}'.format(ep,step))
                break
            state = afterstate
        
    #print(Qvalue)
    if greedy:
        for state in range(N):
            pi[state] = np.argmax(Qvalue[state])   
            
    else:
        inv_tempretures = np.full(N, 1)
        pi = culc_boltzman_policy(Qvalue, inv_tempretures)

    #軌跡の生成
    s_list = [Start]
    s_a_list = []
    score = 0
    
    while not ( (s_list[-1] in Goal) and (min_score <= score <= max_score) ):
        score = 0
        s_list = [Start]
        s_a_list = []
        state = Start
        
        while not (state in Goal):
            if greedy:
                action = choose_a(Qvalue, state, A, epsilon=0)
            else:
                action = choose_a(Qvalue, state, A, epsilon=1.0)
                #action = choose_boltzman_a(pi, state, A)
            s_a_list.append((state, action))
            #行動後の状態
            afterstate = GlidTransition(X,Y,state,Goal,action) #afterstate
            score += R[afterstate]
            s_list.append(afterstate)
            state = afterstate
            
    dummy = 0
    s_a_list.append((state, dummy))
    state_trajectory = np.array(s_list, dtype=int)
    print(state_trajectory)
    print(score)
    
    return [Qvalue, pi, state_trajectory, s_a_list]

def random_make_trajectory_scored(Start, X, Y, Goal, A, R,
                                  min_score, max_score, trunc = 30):
        
    #differ_trans = True
    differ_trans = False
    
    while True:
        s_list = [Start]
        state = Start
        score = 0
        step = 0
        
        while True:
            step += 1
            
            while True:
                action = np.random.randint(0,len(A))
                #行動後の状態
                afterstate = GlidTransition(X,Y,state,Goal,action) #afterstate                
                
                if differ_trans:
                    if not (afterstate is state):
                        break
                else:
                    break
                
            score += R[afterstate]
            s_list.append(afterstate)
            state = afterstate
            
            if step >= trunc or state in Goal:
                break
                
            if min_score <= score <= max_score:
                break

        if min_score <= score <= max_score:
                break

    state_trajectory = np.array(s_list, dtype=int)
    return state_trajectory, score

def random_make_trajectory(Start, X, Y, Goal, A, prob_stop, trunc = 30):
        
    s_list = [Start]
    state = Start
    step = 0
    
    while True:
        step += 1
        action = np.random.randint(0,len(A))

        #行動後の状態
        afterstate = GlidTransition(X,Y,state,Goal,action) #afterstate
        s_list.append(afterstate)
        state = afterstate
        
        if step >= trunc or state in Goal:
            break
        
        if np.random.rand() < prob_stop:
            break

    state_trajectory = np.array(s_list, dtype=int)
    return state_trajectory

def make_failed_trajectory(Start, X, Y, Goal, A, prob_stop, trunc = 30):
    differ_trans = True
    while True:
        s_list = [Start]
        state = Start
        step = 0
    
        while True:
            step += 1
            while True:
                action = np.random.randint(0,len(A))
                #行動後の状態
                afterstate = GlidTransition(X,Y,state,Goal,action) #afterstate                
                
                if differ_trans:
                    if not (afterstate is state):
                        break
                else:
                    break
                
            s_list.append(afterstate)
            state = afterstate
            
            if step >= trunc or state in Goal:
                break
            
            if np.random.rand() < prob_stop:
                break

        if not(state in Goal):
            break
        
    state_trajectory = np.array(s_list, dtype=int)
    return state_trajectory

def make_success_trajectory(Start, X, Y, Goal, A, prob_stop, trunc = 30):
    
    while True:
        s_list = [Start]
        state = Start
        step = 0
    
        while True:
            step += 1
            action = np.random.randint(0,len(A))

            #行動後の状態
            afterstate = GlidTransition(X,Y,state,Goal,action) #afterstate
            s_list.append(afterstate)
            state = afterstate
            
            if step >= trunc or state in Goal:
                break

        if state in Goal:
            break
        
    state_trajectory = np.array(s_list, dtype=int)
    return state_trajectory