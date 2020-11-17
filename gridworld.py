import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

class GridWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape , reward, random_error = None):
        #e.g.) shape = [4,4] , reward = np.zeos(nS)
        
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        nS,nA = np.prod(shape),5
        MAX_Y,MAX_X = shape[0],shape[1]
        P={}
        grid=np.arange(nS).reshape(shape)
        it=np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)} 
            
            #up, down, left, right and stay (total 5 actions)
            ns_up = s if y == 0 else s - MAX_X
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s - 1
            ns_right = s if x == (MAX_X - 1) else s + 1
            ns_stay = s
            
            # No terminal state
            is_done = False

            P[s][UP] = [(1.0, ns_up, reward[s], is_done)]
            P[s][RIGHT] = [(1.0, ns_right, reward[s], is_done)]
            P[s][DOWN] = [(1.0, ns_down, reward[s], is_done)]
            P[s][LEFT] = [(1.0, ns_left, reward[s], is_done)]
            P[s][STAY] = [(1.0, ns_stay, reward[s], is_done)]
            
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