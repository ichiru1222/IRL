import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

act_dict = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0], 4: [0, 0]}

def snum(s, maxY):
    return s[0] + s[1] * maxY #配列から状態に変換

def WallEvaluation(S, s, a):
    """
    s, a : 二次元座標
    S: 座標の集合
    """
    s_dash = s + a
    if not (s_dash == S).all(axis=1).any():
        return s
    return s_dash

class GridWorld(discrete.DiscreteEnv):
    
    """
    Grid World environment
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
        
    o  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  o
    
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape , reward, random_error = None):
        #e.g.) shape = [4,4] , reward = np.zeos(nS)
        
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        
        nS = np.prod(shape) # 状態数
        nA = 5 #行動数

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])
        S = np.array([[i, j] for j in range(MAX_Y) for i in range(MAX_X)])

        while not it.finished:
            
            s = it.iterindex
            y, x = it.multi_index
            
            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA) if s != WallEvaluation(S, s, act_dict[a])} 
            
            
            #up, down, left, right and stay (total 5 actions)
            #ns_up = s if y == 0 else s - MAX_X
            #ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            #ns_left = s if x == 0 else s - 1
            #ns_right = s if x == (MAX_X - 1) else s + 1
            #ns_stay = s
            
            # No terminal state
            is_done = False
            
            for i in P[s]:
                s_next = WallEvaluation(S, s, act_dict[i])
                P[s][i] = [(1.0, snum(s_next, MAX_Y), reward[s], is_done)]
            P[s][4] = [(1.0, snum(s), reward[s], is_done)] 

            #P[s][UP] = [(1.0, ns_up, reward[s], is_done)]
            #P[s][RIGHT] = [(1.0, ns_right, reward[s], is_done)]
            #P[s][DOWN] = [(1.0, ns_down, reward[s], is_done)]
            #P[s][LEFT] = [(1.0, ns_left, reward[s], is_done)]
            #P[s][STAY] = [(1.0, ns_stay, reward[s], is_done)]
                        
            it.iternext()

        'Initial state is state "0" '
        start = 0
        
        isd = np.zeros(nS)
        isd[start] = 1.0

        'Initial state distribution is uniform'
        #isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridWorld, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            S  o  o  o
            o  o  o  o
            o  o  o  o
            o  o  o  o
        where x is your position and T are the two terminal states.
        """
        
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()