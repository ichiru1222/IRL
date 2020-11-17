# -*- coding: utf-8 -*-
"""
@author: Toshiaki
"""

import numpy as np

class ValueIteration:
    
    def __init__(self, env, gamma = 0.99):
        self.gamma = gamma
        self.env = env
        self.P = np.array([[np.eye(1, env.nS, env.P[s][a][0][1])[0] for a in range(env.nA)] for s in range(env.nS)])
        self.reward = np.array([env.P[s][0][0][2] for s in range(env.nS)])
        
    def value_iteration(self):
        
        def expected_value(s, V):
            #compute expected_value of state "s"             
            """
            expected_value = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for s_dash in range(self.env.nS):
                    expected_value[a] += self.P[s][a][s_dash]*V[s_dash]
            """
            expected_value = np.einsum("an, n->a", self.P[s], V) #nはnext_state(s_dash)の意            
            return expected_value
        
        V = np.zeros(self.env.nS)
        pi = np.zeros([self.env.nS, self.env.nA])
        
        theta = 1e-10   
        delta = 0
        
        while True:
            delta = 0
            for s in range(self.env.nS):
                v = V[s].copy()
                V[s] = self.reward[s] + self.gamma*np.max(expected_value(s, V))                                
                delta = max(delta, abs(v-V[s]))
                        
            if delta < theta:
                break
            
        for s in range(self.env.nS):
            greedy_action = np.argmax(expected_value(s, V))
            pi[s][greedy_action] = 1.0 
                
        return V, pi
    
    def get_pi_value(self, pi):
        #piは状態sにおける行動aをとる""確率""となっていることに注意

        def expected_policy_value(s, pi, V):
            #compute expected_value of state "s" 
            """
            expected_policy_value = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for s_dash in range(self.env.nS):
                    expected_policy_value[a] += self.P[s][a][s_dash]*pi[s][a]*V[s_dash]
            """
            expected_policy_value = np.einsum("an, a, n -> a", self.P[s], pi[s], V) #nはnext_state(s_dash)の意
            return expected_policy_value
    
        V_pi = np.zeros(self.env.nS)
        theta = 1e-10
    
        while True:
            delta = 0
            for s in range(self.env.nS):
                v = V_pi[s].copy()
                V_pi[s] = self.reward[s] + self.gamma*np.sum(expected_policy_value(s, pi, V_pi))
               
                delta = max(delta, abs(v-V_pi[s]))
            
            if delta < theta:
                break
            
        return V_pi
        
    def get_state_trajectory(self, env, pi, max_step, noise=0.0):   
        #軌跡の生成
        s = env.reset()
        s_list = []
        s_list.append(s)
        
        while len(s_list) < max_step:
            #epsilon-greedy policy
            if noise > np.random.rand():
                a = np.random.choice(len(pi[s]))
            else:
                a = np.argmax(pi[s])
                
            #行動後の状態
            s_dash, reward, done, info = env.step(a) #afterstate
            s_list.append(s_dash)
            
            if done:
                break
            
            s = s_dash

        trajectory = np.array(s_list, dtype=int)
        #print(trajectory)
        
        return trajectory