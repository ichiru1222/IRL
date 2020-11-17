# -*- coding: utf-8 -*-
"""
MaxEntIRL

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from statistics import mean, median,variance,stdev
import math

def Q_seikika_gap(pi_check,X,Y,V,soft_Q_policy):#ここのpi_checkは軌跡（方策ではない）
    a_num_all=X*Y*5 #対象環境における行動の総数
    
    global Q_list
    Q_list = np.array([])
    for i in range(X*Y):#全状態 #[上，下，左，右，Stay]、P(s,a,a')=1のときの行動価値
        #上
        if i < X:
            Q_list = np.append(Q_list,np.array([0]))
        else:
            Q_list = np.append(Q_list,np.array([V[i-X]]))
        #下
        if i >= X*Y-X:
            Q_list = np.append(Q_list,np.array([0]))
        else:
            Q_list = np.append(Q_list,np.array([V[i+X]]))
        #左    
        if (i % X) == 0:
            Q_list = np.append(Q_list,np.array([0]))
        else:
            Q_list = np.append(Q_list,np.array([V[i-1]]))
        #右    
        if (i % X) == X-1:
            Q_list = np.append(Q_list,np.array([0]))
        else:
            Q_list = np.append(Q_list,np.array([V[i+1]]))
        #stay     
        Q_list = np.append(Q_list,np.array([V[i]]))   

    global Q_list_reshape
    Q_list_reshape = Q_list.reshape((a_num_all,1))
    #print(Q_list_reshape)
    
    global Q_list_seikika
    Q_list_seikika = np.zeros(a_num_all)
    Q_list_reshape = Q_list.reshape(1,a_num_all)
    
    global sort,MEAN_Q,STDEV_Q,MIN_Q,MAX_Q
    sort = sorted(Q_list_reshape[0])
    MEAN_Q = mean(sort[X*4:])
    STDEV_Q = stdev(sort[X*4:])
    MIN_Q = min(sort[X*4:])
    MAX_Q = max(sort[X*4:])
    
    kabelist = []
    
    """
    #標準化なしver
    for s in range(a_num_all):
        #行動価値が0だったら壁→壁リストに入れる
        if not Q_list_reshape[0][s] == 0:
            Q_seikika = Q_list_reshape[0][s]
            Q_list_seikika[s] = Q_seikika
        else:
            kabelist.append(s)
    Q_list_seikika = Q_list_seikika.reshape((X*Y,5))
    """
    
    """
    #標準化ver
    for s in range(125):
        if not Q_list_reshape[0][s] == 0:
            Q_seikika = (Q_list_reshape[0][s]- MEAN_Q)/STDEV_Q
            Q_list_seikika[s] = Q_seikika
        else:
            kabelist.append(s)
    Q_list_seikika = Q_list_seikika.reshape((25,5))
    """
    
    #[0,1]正規化ver
    for s in range(125):
        if not Q_list_reshape[0][s] == 0:
            Q_seikika = (Q_list_reshape[0][s]- MIN_Q)/(MAX_Q - MIN_Q)
            Q_list_seikika[s] = Q_seikika
        else:
            kabelist.append(s)
    Q_list_seikika = Q_list_seikika.reshape((25,5))
    
    
    grid_mean = []
    for r in range(len(Q_list_seikika)):
        wa,count = 0,0
        for d in range(5):
            if Q_list_seikika[r][d] != -10:
                wa += Q_list_seikika[r][d]
                count += 1
        grid_mean.append(wa/count)
    q_gap_one_list = []
    state = 20
    q_gap_one = 0
    
    #各マスの期待値計算
    global Q_kitaiti
    Q_kitaiti = [0]*X*Y
    for i in range(X*Y):
        for w in range(5):
            Q_kitaiti_a = Q_list_seikika[i][w]*soft_Q_policy[i][w]
            Q_kitaiti[i] += Q_kitaiti_a
    
    for i in range((len(pi_check)-1)):
        next_state = pi_check[i+1]
        #期待値版でやってみる
        if next_state == state-X:
            q_gap_one = Q_kitaiti[state] - Q_list_seikika[state][0]
        elif next_state == state+X:
            q_gap_one = Q_kitaiti[state] - Q_list_seikika[state][1]
        elif next_state == state-1:
            q_gap_one = Q_kitaiti[state] - Q_list_seikika[state][2]
        elif next_state == state+1:
            q_gap_one = Q_kitaiti[state] - Q_list_seikika[state][3]
        elif next_state == state:
            q_gap_one = Q_kitaiti[state] - Q_list_seikika[state][4]
        if state == 4:
            q_gap_one = 0
        state = next_state
        q_gap_one_list.append(q_gap_one)        
    
    return q_gap_one_list
    
def Q_gap_sum_list(q_gap_one_list):
    q_gap_sum_list = []
    q_gap_sum = 0
    for i in range(len(q_gap_one_list)): 
        q_gap_sum = q_gap_sum + q_gap_one_list[i]
        q_gap_sum_list.append(q_gap_sum)
    return q_gap_sum_list

######################MAXENT#########################################################
### Φ(s)の計算 ######################################
def phi(state, x_size, y_size):
    #one-hotベクトル化する
    phi_s = np.zeros(x_size*y_size)
    for i in range(x_size*y_size):
        if i == state:
            phi_s[i] = 1
        else:
            phi_s[i] = 0 
    #行列で返す
    return phi_s 

def Mu(traj, x_size, y_size):
    Mu_s = np.zeros(x_size*y_size)
    for s in traj:
        Mu_s = Mu_s + phi(s, x_size, y_size)        
    return Mu_s

def MuE(trajectories, x_size, y_size):
    MuE_m = np.zeros(x_size*y_size)
    
    for traj in trajectories:
        MuE_m += Mu(traj, x_size, y_size)
    
    MuE_m = MuE_m / len(trajectories)
    
    return MuE_m

def MaxEntIRL_GridWorld(env, trajectories, delta, max_step, learning_rate):#MaxEnt本体
    P = np.array([[np.eye(1, env.nS, env.P[s][a][0][1])[0] for a in range(env.nA)] for s in range(env.nS)])
    print(P)
    x_size,y_size = env.shape[0],env.shape[1]
    
    global muE
    muE = MuE(trajectories, x_size, y_size)
    #muE[4],muE[24] = 0.5,0.5
    #print(muE)    
    theta = np.random.uniform(-0.5, 0.5, size=env.nS)
    feature_matrix = np.eye(env.nS)

    R = np.dot(theta, feature_matrix.T)    
    print("initial reward")
    print(R)    
    #for i in tqdm(range(n_epochs)):

    norm_grad = float('inf')
    while(norm_grad > delta):  
        #Rの計算
        R = np.dot(theta, feature_matrix.T)    

        #インナーループ
        """Backward pass：後ろからたどる"""
        policy = np.zeros([env.nS, env.nA])

        Z_a = np.zeros([env.nS, env.nA])
        Z_s = np.ones([env.nS])
        
        #Note:N回のイテレーションの”N”は，軌跡の長さ
        for n in range(max_step):            
            Z_a = np.einsum("san, s, n -> sa", P, np.exp(R), Z_s) #nはnext_stateの意
            Z_s = np.sum(Z_a, axis = 1) #Z_sの初期化位置は"ここ"
                        
        policy = np.einsum("sa, s -> sa", Z_a, 1/Z_s)#各状態における行動選択確率：：：これがsoft_Q_policy
        
        """Forward pass"""
        Dt = np.zeros([max_step, env.nS]) #論文アルゴリズム中の Dを指す
         
        #initialize mu[0] based on trajectories initial state
        for trajectory in trajectories:
            Dt[0][trajectory[0]] += 1
        Dt /= len(trajectories)
                
        for t in range(1, max_step):
            Dt[t] = np.einsum("s, sa, san -> n", Dt[t-1], policy, P) 
        Ds = Dt.sum(axis=0)#頻度
        
        #print(mu)
        #print(feature_matrix.T.dot(mu))
        #grad = muE - feature_matrix.T.dot(mu)
        
        #L2ノルム
        grad = muE - feature_matrix.T.dot(Ds)
        norm_grad = np.linalg.norm(grad, ord=2)

        #print(grad)
        #if np.random.rand() < 0.0001:
        f,i=math.modf(norm_grad*100)
        #print(f)
        if f < 0.001:
            print(norm_grad) 
        theta += learning_rate * grad #最大化問題なので勾配降下が勾配上昇（勾配を加える）になっていることに注意
        #print(theta)
                
    print("MaxEntIRL ended.")
    print(R)
    return R, policy
    
def if_true_color_red(val, else_color):
    if val:
        return 'r'
    else:
        return else_color

def miyasui_plot(for_plot,X,Y):#Q値を可視化
    max_bool = for_plot == np.max(for_plot, axis=0)
    max_color_k = np.vectorize(if_true_color_red)(max_bool,'k')
    #max_color_w = np.vectorize(if_true_color_red)(max_bool,'w')
    for_plot_array_round = np.round(for_plot, decimals=2)
    # 行動価値関数を表示
    ax = plt.gca()
    plt.xlim(0,X)
    plt.ylim(0,Y)
    ax.xaxis.set_ticklabels([])#x軸の数字を消す
    ax.yaxis.set_ticklabels([])#y軸の数字を消す
    
    for i in range(X):
        for j in range(Y):
            # rect
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            #rect2 = plt.Rectangle(xy =(i+0.37,j+0.4) , width=0.25,height=0.2,edgecolor='black',facecolor='white',alpha=1)
            ax.add_patch(rect)
            #ax.add_patch(rect2)
            # diag
            diag = plt.Line2D(xdata=(i,i+1), ydata=(j,j+1),color='k',linewidth=.5)
            ax.add_line(diag)
            diag = plt.Line2D(xdata=(i,i+1), ydata=(j+1,j),color='k',linewidth=.5)
            ax.add_line(diag)
            # 座標のインデックスの調整
            x = -j-1 
            y = i
            # text
            plt.text(i+ 0.75, j+0.45, "%s" % (str(for_plot_array_round[0,x,y])), color=max_color_k[0,x,y])
            plt.text(i+ 0.4, j+0.8, "%s" % (str(for_plot_array_round[1,x,y])), color=max_color_k[1,x,y])
            plt.text(i+ 0.025, j+0.45, "%s" % (str(for_plot_array_round[2,x,y])), color=max_color_k[2,x,y])
            plt.text(i+ 0.4, j+0.1, "%s" % (str(for_plot_array_round[3,x,y])), color=max_color_k[3,x,y])
            plt.text(i+ 0.4, j+0.45, "%s" % (str(for_plot_array_round[4,x,y])), color=max_color_k[4,x,y])
    plt.show()

def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]
        
############ ここまで関数 ########### ここからmain ##############

if __name__ == '__main__':
    import gridworld
    from value_iteration import ValueIteration
    
    #setting env
    X,Y = 5,5
    #setting reward
    grid_shape =[X,Y]
    reward = np.full(np.prod(grid_shape), 0.0)
    #setting expert 
    env = gridworld.GridWorld(grid_shape, reward)
    gamma = 0.9
    #0.99,0.95,0.90,0.85,0.80
    #traj =[[20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 15, 10, 5, 6, 7, 8, 3, 4], [20, 15, 16, 11, 6, 7, 8, 9, 4], [20, 21, 16, 11, 6, 7, 8, 9, 4], [20, 15, 16, 17, 12, 13, 14, 9, 4], [20, 21, 22, 17, 12, 7, 8, 3, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 3, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 21, 16, 11, 12, 7, 2, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 22, 17, 18, 13, 14, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 15, 10, 11, 12, 7, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 10, 11, 12, 7, 2, 3, 4], [20, 21, 22, 23, 18, 13, 14, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 9, 4], [20, 15, 10, 11, 12, 13, 8, 9, 4], [20, 15, 16, 17, 18, 13, 14, 9, 4], [20, 21, 22, 17, 12, 13, 8, 3, 4], [20, 15, 16, 17, 18, 13, 8, 3, 4], [20, 21, 22, 17, 12, 7, 8, 3, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 11, 12, 13, 8, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 21, 22, 17, 18, 13, 14, 9, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 15, 16, 11, 12, 13, 8, 3, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 21, 22, 17, 12, 13, 8, 3, 4], [20, 15, 16, 17, 18, 13, 8, 9, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 15, 10, 11, 12, 7, 8, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 15, 10, 11, 6, 1, 2, 3, 4], [20, 15, 10, 5, 6, 7, 8, 3, 4], [20, 21, 16, 11, 6, 7, 2, 3, 4], [20, 15, 10, 11, 6, 7, 8, 9, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 21, 22, 23, 18, 13, 8, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 17, 12, 13, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 12, 13, 14, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 15, 10, 11, 6, 7, 8, 9, 4], [20, 21, 22, 23, 18, 13, 8, 9, 4], [20, 21, 16, 17, 12, 13, 14, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 10, 11, 6, 7, 8, 3, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 18, 13, 8, 9, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 15, 10, 11, 12, 13, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 21, 16, 17, 12, 7, 8, 9, 4], [20, 15, 16, 17, 18, 13, 8, 3, 4], [20, 21, 16, 11, 12, 7, 2, 3, 4], [20, 21, 16, 11, 12, 13, 8, 3, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 16, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 12, 13, 8, 9, 4], [20, 21, 22, 17, 12, 7, 8, 9, 4], [20, 21, 22, 17, 18, 13, 8, 3, 4], [20, 21, 16, 11, 12, 7, 8, 9, 4], [20, 21, 22, 17, 12, 13, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 15, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 17, 18, 13, 8, 9, 4], [20, 21, 22, 17, 12, 7, 8, 9, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4]]    
    traj=[[20, 21, 22, 23, 18, 13, 8, 9, 4], [20, 21, 16, 17, 18, 13, 14, 9, 4], [20, 15, 10, 11, 12, 7, 2, 3, 4], [20, 15, 16, 17, 12, 13, 8, 3, 4], [20, 21, 16, 11, 12, 7, 8, 3, 4], [20, 15, 16, 17, 12, 7, 8, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 10, 5, 6, 7, 2, 3, 4], [20, 21, 16, 11, 6, 7, 2, 3, 4], [20, 21, 22, 23, 18, 13, 8, 9, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 21, 16, 17, 18, 13, 14, 9, 4], [20, 15, 10, 11, 12, 13, 14, 9, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 11, 12, 13, 14, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 21, 16, 17, 12, 7, 2, 3, 4], [20, 15, 16, 17, 12, 13, 14, 9, 4], [20, 21, 16, 17, 18, 13, 8, 9, 4], [20, 21, 16, 17, 18, 13, 8, 9, 4], [20, 21, 16, 17, 12, 13, 14, 9, 4], [20, 15, 10, 11, 6, 1, 2, 3, 4], [20, 15, 16, 11, 6, 1, 2, 3, 4], [20, 21, 22, 17, 12, 13, 8, 3, 4], [20, 21, 22, 23, 18, 13, 8, 3, 4], [20, 15, 16, 11, 6, 1, 2, 3, 4], [20, 21, 16, 11, 12, 13, 14, 9, 4], [20, 15, 16, 11, 6, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 21, 16, 11, 6, 7, 2, 3, 4], [20, 21, 16, 17, 18, 13, 8, 9, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 15, 10, 5, 6, 7, 8, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 15, 16, 11, 6, 7, 8, 9, 4], [20, 21, 16, 17, 12, 7, 8, 3, 4], [20, 15, 10, 11, 6, 7, 8, 9, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 21, 16, 17, 12, 13, 14, 9, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 10, 11, 6, 7, 2, 3, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 10, 5, 0, 1, 2, 3, 4], [20, 15, 16, 11, 6, 1, 2, 3, 4], [20, 21, 22, 17, 18, 13, 14, 9, 4], [20, 21, 22, 17, 18, 13, 8, 3, 4], [20, 15, 10, 11, 12, 13, 8, 3, 4], [20, 15, 10, 11, 12, 13, 8, 9, 4], [20, 15, 10, 5, 6, 7, 8, 9, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 16, 11, 6, 1, 2, 3, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 15, 10, 11, 6, 7, 8, 9, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 15, 10, 11, 6, 1, 2, 3, 4], [20, 15, 16, 11, 12, 7, 2, 3, 4], [20, 15, 16, 17, 18, 13, 8, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 15, 16, 11, 12, 7, 2, 3, 4], [20, 21, 22, 17, 18, 13, 8, 3, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 21, 22, 17, 12, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 21, 16, 11, 6, 1, 2, 3, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 10, 5, 6, 1, 2, 3, 4], [20, 21, 22, 23, 18, 19, 14, 9, 4], [20, 21, 16, 17, 12, 13, 8, 9, 4], [20, 21, 16, 17, 18, 19, 14, 9, 4], [20, 15, 16, 11, 12, 13, 14, 9, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 16, 17, 12, 7, 2, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 21, 22, 17, 12, 13, 8, 9, 4], [20, 21, 22, 17, 12, 13, 8, 9, 4], [20, 15, 10, 5, 6, 7, 8, 3, 4], [20, 15, 10, 11, 6, 1, 2, 3, 4], [20, 21, 22, 23, 24, 19, 14, 9, 4], [20, 15, 16, 11, 12, 7, 2, 3, 4], [20, 21, 22, 17, 18, 19, 14, 9, 4], [20, 21, 16, 17, 18, 13, 8, 9, 4], [20, 21, 16, 11, 6, 7, 8, 9, 4], [20, 15, 16, 17, 18, 19, 14, 9, 4], [20, 15, 16, 17, 12, 7, 8, 3, 4]]
    traj_check = [[20,21,22,23,24,19,18,17,16,15,10,11,12,13,14,9,8,7,6,5,0,1,2,3,4]]
    num_traj = len(traj)
    max_step = len(traj[0])
    
    """reward estimation"""
    #delta は　勾配ベクトルにおけるL2ノルムの閾値（deltaを下回ったら推定完了とする）,learning_lateは勾配変化の学習率
    delta, learning_rate = 0.01, 0.015
    est_reward, soft_Q_policy = MaxEntIRL_GridWorld(env, traj, delta, max_step, learning_rate)                
    
    for i in range(X*Y):#全状態 #[上，下，左，右，Stay]、P(s,a,a')=1のときの行動価値
        #上
        if i < X:
            soft_Q_policy[i][4] = soft_Q_policy[i][4] + soft_Q_policy[i][0] 
            soft_Q_policy[i][0] = 0
        #下
        if i >= X*Y-X:
            soft_Q_policy[i][4] = soft_Q_policy[i][4] + soft_Q_policy[i][1] 
            soft_Q_policy[i][1] = 0
        #左    
        if (i % X) == 0:
            soft_Q_policy[i][4] = soft_Q_policy[i][4] + soft_Q_policy[i][2] 
            soft_Q_policy[i][2] = 0
        #右    
        if (i % X) == X-1:
            soft_Q_policy[i][4] = soft_Q_policy[i][4] + soft_Q_policy[i][3] 
            soft_Q_policy[i][3] = 0 
    
    #報酬を保存
    np.savetxt("R_X5Y5.csv",est_reward.reshape((X,Y)),delimiter=", ")
    #print(est_reward)
    
    env_est = gridworld.GridWorld(grid_shape, est_reward)
    est_agent = ValueIteration(env_est, gamma)
    
    #状態価値を出す#確率的な方策にも対応
    V_est = est_agent.get_pi_value(soft_Q_policy)
    print(V_est)
    #np.savetxt("V_Pro_1.csv",V_est.reshape((5,5)),delimiter=", ")
    
    gap_sum_dist = []
    for q in range(len(traj)):
        pi_check = traj[q]
        #ここのpi_checkは軌跡（方策ではない）
        #gapをだす
        q_gap_one_list = Q_seikika_gap(pi_check,X,Y,V_est,soft_Q_policy)
        q_gap_sum_list = Q_gap_sum_list(q_gap_one_list)
        #print(q_gap_sum_list[-1])
        gap_sum_dist.append(q_gap_sum_list[-1])
    gap_sum_dist.sort()
    #plt.figure()
    last_mean = np.mean(gap_sum_dist)
    #plt.axvline(last_mean, ls = "--", color = "navy")
    #plt.savefig('figure5.png')
 
 ###############################miyasui_plot#################################################
    for_plot_a =[]
    #direction = {'stay':1, 'migi':2, 'hidari':3, 'sita':4, 'ue':5}
    stay, migi, hidari, sita, ue =[],[],[],[],[]
    list_k = Q_list_seikika.reshape((1,X*Y*5)) #Q値
    for i in range(X*Y):
        stay.append(list_k[0][5*(i+1)-1])
    for i in range(X*Y):
        migi.append(list_k[0][5*(i+1)-2])
    for i in range(X*Y):
        hidari.append(list_k[0][5*(i+1)-3])
    for i in range(X*Y):
        sita.append(list_k[0][5*(i+1)-4])
    for i in range(X*Y):
        ue.append(list_k[0][5*(i+1)-5])
                
    migi_s, ue_s, hidari_s, sita_s, stay_s = list(split_list(migi,X)), list(split_list(ue,X)) , list(split_list(hidari,X)), list(split_list(sita,X)), list(split_list(stay,X))
    migi_all,ue_all,hidari_all,sita_all,stay_all = [],[],[],[],[]
    
    for i in range(X):
        migi_all.append(migi_s[i])
        hidari_all.append(hidari_s[i])
        ue_all.append(ue_s[i])
        sita_all.append(sita_s[i])
        stay_all.append(stay_s[i])
    for_plot_a.append(migi_all)
    for_plot_a.append(ue_all)
    for_plot_a.append(hidari_all)
    for_plot_a.append(sita_all)
    for_plot_a.append(stay_all)
    
    plt.figure()
    miyasui_plot(for_plot_a,X,Y)
    
    
    for_plot_a =[]
    #direction = {'stay':1, 'migi':2, 'hidari':3, 'sita':4, 'ue':5}
    stay, migi, hidari, sita, ue =[],[],[],[],[]
    list_k = soft_Q_policy.reshape((1,125)) #最適方策
    for i in range(X*Y):
        stay.append(list_k[0][5*(i+1)-1])
    for i in range(X*Y):
        migi.append(list_k[0][5*(i+1)-2])
    for i in range(X*Y):
        hidari.append(list_k[0][5*(i+1)-3])
    for i in range(X*Y):
        sita.append(list_k[0][5*(i+1)-4])
    for i in range(X*Y):
        ue.append(list_k[0][5*(i+1)-5])
                
    migi_s, ue_s, hidari_s, sita_s, stay_s = list(split_list(migi,X)), list(split_list(ue,X)) , list(split_list(hidari,X)), list(split_list(sita,X)), list(split_list(stay,X))
    migi_all,ue_all,hidari_all,sita_all,stay_all = [],[],[],[],[]
    
    for i in range(X):
        migi_all.append(migi_s[i])
        hidari_all.append(hidari_s[i])
        ue_all.append(ue_s[i])
        sita_all.append(sita_s[i])
        stay_all.append(stay_s[i])
    
    for_plot_a.append(migi_all)
    for_plot_a.append(ue_all)
    for_plot_a.append(hidari_all)
    for_plot_a.append(sita_all)
    for_plot_a.append(stay_all)
    
    plt.figure()
    miyasui_plot(for_plot_a,X,Y)
    
########################################################################################
    
    #sum_list_listは全部の動線のThの経過が入る
    sum_list_list, gap_sum_dist2 = [],[]
    for q in range(len(traj_check)):
        pi_check = traj_check[q]
        q_gap_one_list = Q_seikika_gap(pi_check,X,Y,V_est,soft_Q_policy)#1個体のT^h_tの経過
        q_gap_sum_list = Q_gap_sum_list(q_gap_one_list)#1個体のT^h
        sum_list_list.append(q_gap_sum_list)#全個体のT^h_tの経過のリスト
        gap_sum_dist2.append(q_gap_sum_list[-1])#全個体のT^hのリスト
        
    #経過
    plt.figure()
    plt.axhline(y=0,ls = "--",color="green")
    for i in range(len(sum_list_list)):
        plt.plot([x for x in range(0,len(sum_list_list[i]))],sum_list_list[i],label = "{}".format(i+1))
        plt.xticks([x for x in range(0,len(sum_list_list[i]))])
        plt.grid(b=None)
        plt.xlabel("t")
        plt.ylabel("Sus^h")
        plt.ylim(-2.5,2.5)
        plt.grid(color='gray')

####################################################################################
def NextRand(state,X,Y):
    next_state = 0
    option = np.array([state-X,state+X,state-1,state+1,state])#[上、下、左、右、stay]
    if state == 0:#1
        next_state = next_state = np.random.choice(option,p=[0,0.33,0,0.33,0.34])
    elif (0 < state < X):#2
        next_state = np.random.choice(option,p=[0,0.25,0.25,0.25,0.25])
    elif ((state % X) == 0) and (state != 0) and (state != X*X-X):#4
        next_state = np.random.choice(option,p=[0.25,0.25,0,0.25,0.25])
    elif ((state % X) == X-1) and (state != X-1) and (state != X*X-1):#6
        next_state = np.random.choice(option,p=[0.25,0.25,0.25,0,0.25])
    elif state == X*X-X:#7
        next_state = np.random.choice(option,p=[0.33,0,0,0.33,0.34])
    elif (X*X-X < state < X*X-1):#8
        next_state = np.random.choice(option,p=[0.33,0,0,0.33,0.34]) 
    elif state == X*X-1:#9
        next_state = np.random.choice(option,p=[0.33,0,0.33,0,0.34]) 
    else:#5
        next_state = np.random.choice(option,p=[0.2,0.2,0.2,0.2,0.2])
    if state == X-1:#3
        next_state = np.random.choice(option,p=[0,0,0,0,1])
        
    return next_state

def NextRand2(state,X,Y):
    next_state = 0
    option = np.array([state-X,state+X,state-1,state+1,state])#[上、下、左、右、stay]
    if state == 0:#1
        next_state = next_state = np.random.choice(option,p=[0,0.5,0,0.5,0])
    elif (0 < state < X):#2
        next_state = np.random.choice(option,p=[0,0.33,0.33,0.34,0])
    elif ((state % X) == 0) and (state != 0) and (state != X*X-X):#4
        next_state = np.random.choice(option,p=[0.33,0.33,0,0.34,0])
    elif ((state % X) == X-1) and (state != X-1) and (state != X*X-1):#6
        next_state = np.random.choice(option,p=[0.33,0.33,0.34,0,0])
    elif state == X*X-X:#7
        next_state = np.random.choice(option,p=[0.5,0,0,0.5,0])
    elif (X*X-X < state < X*X-1):#8
        next_state = np.random.choice(option,p=[0.5,0,0,0.5,0]) 
    elif state == X*X-1:#9
        next_state = np.random.choice(option,p=[0.5,0,0.5,0,0]) 
    else:#5
        next_state = np.random.choice(option,p=[0.25,0.25,0.25,0.25,0])
    if state == X-1:#3
        next_state = np.random.choice(option,p=[0,0,0,0,1])
    return next_state

def TrajmakerXX(X,Y,len_traj,traj_num):#X*Yの最短経路をランダムに選択する、traj_numは軌跡数
    traj=[]
    while(True):
        state = X*(X-1)
        traj_one = [X*(X-1)]
        while(True):
            next_state = NextRand(state,X,Y)
            traj_one.append(next_state)
            state = next_state
            if state == X-1:
                break
        if len(traj_one) < len_traj:#最短で(Y-1)+(X-1)
            traj.append(traj_one)
            
        if len(traj) == traj_num:
            break
    return traj

def TrajmakerINF(X,Y,traj_num):#X*Yの最短経路をランダムに選択する、traj_numは軌跡数
    traj=[]
    while(True):
        state = X*(X-1)
        traj_one = [X*(X-1)]
        while(True):
            next_state = NextRand(state,X,Y)
            traj_one.append(next_state)
            state = next_state
            #print(state)
            if state == X-1:
                break
        traj.append(traj_one)
        #print(len(traj))
            
        if len(traj) == traj_num:
            break
    return traj
"""
#ここから
traj_sum0 = []
while(True):
    traj_ep=TrajmakerINF(X,Y,1)#軌跡一つ生成
    #traj_ep=TrajmakerXX(X,Y,25,1)#軌跡一つ生成
    #traj_ep=[[20, 15, 15, 16, 16, 11, 6, 7, 8, 9, 4]]
    state = 20
    gap_one,gap_sum_ep = 0,0
    for i in range((len(traj_ep[0])-1)):
        next_state = traj_ep[0][i+1]
        if next_state == state-X:
            gap_one = Q_list_seikika[state][0]-MEAN_Q
        elif next_state == state+X:
            gap_one = Q_list_seikika[state][1]-MEAN_Q
        elif next_state == state-1:
            gap_one = Q_list_seikika[state][2]-MEAN_Q
        elif next_state == state+1:
            gap_one = Q_list_seikika[state][3]-MEAN_Q
        elif next_state == state:
            gap_one = Q_list_seikika[state][4]-MEAN_Q
        if state == 4:
            gap_one = 0
        state = next_state
        gap_sum_ep += gap_one
        #print(gap_one)
    if (-1 < gap_sum_ep < 1):
    #if gap_sum_ep == 0:#理想
        traj_sum0.append(traj_ep[0])
        print(len(traj_sum0))
        print(gap_sum_ep)
    if len(traj_sum0) == 100:
        break
print(traj_sum0)

y_last_list = []
y_min=0
for i in range(len(traj_sum0)):
    gap_ep = Q_seikika_gap(traj_sum0[i],X,Y,V_est)#1個体のT^h_tの経過
    gap_sum_ep = Q_gap_sum_list(gap_ep)#1個体のT^h
    y_last = logistic(gap_sum_ep[-1],a,k,x0)
    y_last_list.append(y_last)
y_min=min(y_last_list)
print("y_last_list={}".format(y_last_list))
print("y_min={}".format(y_min))
print("y_mean={}".format(mean(y_last_list)))
"""
