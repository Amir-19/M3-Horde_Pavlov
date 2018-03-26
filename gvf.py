import numpy as np
from utils import *

class GVF:

    def __init__(self, num_state, lam = 0.9, alpha = 0.1/1, beta = 0.001/1, is_offpolicy = False, bhv_policy = None, target_policy = None):

        # GVF answer part parameters
        self.num_state = num_state  # num of states or number of features
        self.is_offpolicy = is_offpolicy
        self.alpha = alpha  # TD step size
        self.lam = lam  # lambda for eligibility traces
        self.e = np.zeros(num_state)  # eligibility traces
        self.theta = np.zeros(num_state)  # weights for the TD learning

        # GVF answer part parameters (if off-policy)
        if self.is_offpolicy:
            self.w = np.zeros(num_state)  # second set of weights for GTD
            self.beta = beta  # step size for w in GTD
            self.target_policy = target_policy  # target policy for off-policy learning
            self.bhv_policy = bhv_policy  # behavior policy for off-policy learning

        # RUPEE and UDE setup
        self.rupee = RUPEE(num_state, (1-lam)*alpha/30, 5*alpha)
        self.ude = UDE(0.5*alpha)
        self.rv = None
        self.uv = None
        # keeping state and last state
        self.state = None
        self.delta = 0

        # storage
        self.cum = None
        self.sdelta = None
    def gtd(self, c, gamma, state_prime, action):

        delta = c + gamma * (np.dot(self.theta.T, state_prime)) - (
                np.dot(self.theta.T, self.state))
        if type(delta) is np.ndarray:
            delta = delta[0]
        self.sdelta = delta
        self.cum = c
        rho = self.target_policy(state = state_prime,action = action) / self.bhv_policy(state = state_prime,action = action)
        self.e = rho * (self.state + gamma * self.lam * self.e)
        self.theta = self.theta + self.alpha * (np.dot(delta, self.e) - state_prime * (1 - self.lam) * np.dot(self.e.T, self.w) * state_prime)
        self.w = self.w + self.beta * (np.dot(delta, self.e) - np.dot(self.state.T, self.w) * self.state)

        # RUPEE and UDE step
        self.rupee.update(delta,self.e,self.state)
        self.uv = self.ude.update(delta)

        # go to the next state
        self.delta = delta
        self.state = state_prime

    def td(self, c , gamma, state_prime):


        delta = c + gamma * (np.dot(self.theta.T, state_prime)) - (np.dot(self.theta.T, self.state))
        if type(delta) is np.ndarray:
            delta = delta[0]
        self.sdelta = delta
        self.cum = c
        self.e = self.state + gamma * self.lam * self.e
        self.theta = self.theta +self. alpha * (np.dot(delta, self.e))


        # RUPEE and UDE step
        self.rupee.update(delta,self.e,self.state)
        self.uv = self.ude.update(delta)

        # go to the next state
        self.delta = delta
        self.state = state_prime

    def get_prediction(self,state):

        return self.theta[state]

    def set_initial_state(self, state):

        self.state = state

