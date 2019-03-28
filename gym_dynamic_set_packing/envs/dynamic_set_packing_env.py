import gym
from gym import spaces
import numpy as np
import random
from ..matchers import GurobiMatcher
import os
import csv

class DynamicSetPackingBinaryEnv(gym.Env):

    """
    Abstract class representing a dynamic set packing problem with 0/1 matching.
    """
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
                low=np.zeros(state_dim),
                high=np.full(state_dim, np.inf), dtype=np.float32)
        self.seed()

        self.state = None
        self.reset()


    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0.0
        if action == 1:
            chosen_match = self._perform_match(self.state)
            reward = self._run_match(chosen_match)


        # elements arrive and depart
        self._arrive_and_depart()

        # do we need these dummy values?
        info = {'no info': 'no info'}
        done = False

        return self.state, reward, done, info

    ## the following MUST be implemented in the child
    def _run_match(self, match):
        """Must be overridden by a child. Takes the output of the match solver
        and uses it to update the state, possibly with some probability of
        failure per match type. Returns the reward from the match."""

        raise NotImplementedError

    def _perform_match(self, state):
        "Takes a state and finds a match to return. Does not actually modify state. Must be provided by child."
        raise NotImplementedError

    def reset(self):
        """Initializes the environment. Must be provided by child."""
        raise NotImplementedError

    ## the following MAY be overridden by a child
    def _arrive_and_depart(self):
        "Simualtes arrival and departure of elements. May be overridden by child."
        pass

    def render(self, mode='human', close=False):
        "Just prints out the current state vector. May be overriden by child if desired."
        return self.state

    def seed(self):
        random.seed(1)
        np.random.seed(1)

class AdversarialEnv(DynamicSetPackingBinaryEnv):
    def __init__(self):
        super(AdversarialEnv, self).__init__(3)
        feasible_sets = np.array([[1.0,1.0],
                                  [1.0,1.0],
                                  [1.0,0.0]])

        self.matcher = GurobiMatcher(feasible_sets)

    def reset(self):
        self.state = np.zeros(self.state_dim, dtype=np.float32)

        return self.state

    def _perform_match(self, state):
        return self.matcher.match(state)

    def _run_match(self, match):
        total_match = self.matcher.valid_sets @ match
        match_cardinality = np.sum(total_match * np.array([1.0,1.0,50.0]))
        self.state = self.state - (total_match.astype('float32'))
        return match_cardinality

    def _arrive_and_depart(self):
        arrivals = np.zeros(3)
        if np.random.rand() > 0.5:
            arrivals[0] += 1
        if np.random.rand() > 0.7:
            arrivals[1] += 1
        if np.random.rand() > 0.9:
            arrivals[2] += 1
        if self.state[2] > 0:
            if np.random.rand() > 0.1:
                self.state[2] = 0

        self.state += arrivals

class GurobiBinaryEnv(DynamicSetPackingBinaryEnv):
    "A simple test environment that uses Gurobi to find a maximal match."
    def __init__(self):
        super(GurobiBinaryEnv, self).__init__(16) # has to be hard coded :(
        filename = os.path.join(os.path.dirname(__file__), 'bloodTypeVectors.csv')
        feasible_sets = np.genfromtxt(filename,skip_header=1, delimiter=',').transpose()
        self.matcher = GurobiMatcher(feasible_sets)

    ## required overrides
    def reset(self):
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        #self.state[0] = 8
        #self.state[4] = 8
        return self.state

    def _perform_match(self, state):
        return self.matcher.match(state)

    def _run_match(self, match):
        # match is an array of weights for self.matcher.valid_sets
        total_match = self.matcher.valid_sets @ match
        match_cardinality = np.sum(total_match)
        self.state = self.state - (total_match.astype('float32'))
        return match_cardinality

    # optional override
    def _arrive_and_depart(self):
        # arrive
        for i in range(len(self.state)):
            if np.random.rand() > 0.5:
                self.state[i] += 1
            if np.random.rand() > 0.3:
                if self.state[i] > 0:
                    self.state[i] -= 1

class SillyTestEnv(DynamicSetPackingBinaryEnv):
    "A very silly test environment. More to test whether the code runs than anything realistic."

    def __init__(self):
        super(SillyTestEnv, self).__init__(16)

    ## required overrides
    def reset(self):
        self.state = np.ones(self.state_dim, dtype=np.float32)
        return self.state

    def _perform_match(self, state):
        # just return the whole state -- everything always gets matched
        return self.state

    def _run_match(self, match):
        match_cardinality = np.sum(match)
        self.state = self.state - match.astype('float32')
        # self.state should always be 0 here
        return match_cardinality

    def _arrive_and_depart(self):
        # arrive
        for i in range(len(self.state)):
            if np.random.rand() > 0.5:
                self.state[i] += 1
            if np.random.rand() > 0.3:
                if self.state[i] > 0:
                    self.state[i] -= 1
