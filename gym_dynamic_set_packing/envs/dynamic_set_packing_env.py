import gym
from gym import spaces
import numpy as np
import random
from ..matchers import GurobiMatcher, GurobiWeightedMatcher
import os
import csv

class DynamicSetPackingTypeWeight(gym.Env):
    """
    Abstract class representing a dynamic set packing problem with type
    weight action space.
    """
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.action_space = spaces.Box(
                low=np.full(state_dim, -np.inf),
                high=np.full(state_dim, np.inf), dtype=np.float32)

        self.observation_space = spaces.Box(
                low=np.zeros(state_dim),
                high=np.full(state_dim, np.inf), dtype=np.float32)
        self.seed()

        self.state = None
        self.reset()


    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0.0
        
        chosen_match = self._perform_match(self.state, action)
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

    def _perform_match(self, state, action):
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

class GurobiWeightedEnv(DynamicSetPackingTypeWeight):
    "A simple test environment that uses Gurobi to find a maximal match."
    def __init__(self):
        super(GurobiWeightedEnv, self).__init__(16) # has to be hard coded :(
        filename = os.path.join(os.path.dirname(__file__), 'bloodTypeVectors.csv')
        feasible_sets = np.genfromtxt(filename,skip_header=1, delimiter=',').transpose()
        self.matcher = GurobiWeightedMatcher(feasible_sets)

    ## required overrides
    def reset(self):
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        #self.state[0] = 8
        #self.state[4] = 8
        return self.state

    def _perform_match(self, state, action):
        return self.matcher.match(state, action)

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
        super(AdversarialEnv, self).__init__(5)
        feasible_sets = np.array([[1.0,1.0],
                                  [1.0,1.0],
                                  [1.0,0.0],
                                  [1.0,0.0],
                                  [1.0,0.0]])
        self.time_step = 0

        self.matcher = GurobiMatcher(feasible_sets)

    def reset(self):
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.time_step = 0

        return self.state

    def _perform_match(self, state):
        return self.matcher.match(state)

    def _run_match(self, match):
        total_match = self.matcher.valid_sets @ match
        match_cardinality = np.sum(total_match)
        self.state = self.state - (total_match.astype('float32'))
        return match_cardinality

    def _arrive_and_depart(self):
        if self.time_step == 0:
            self.state = np.array([1.0,1.0,0.0,0.0,0.0])
            self.time_step = 1
        elif self.time_step == 1:
            self.state += np.array([0.0,0.0,1.0,1.0,1.0])
            self.time_step = 2
        elif self.time_step == 2:
            self.state = np.zeros(5)
            self.time_step = 0

        

class GurobiBinaryEnv(DynamicSetPackingBinaryEnv):
    "A simple test environment that uses Gurobi to find a maximal match."
    def __init__(self):
        super(GurobiBinaryEnv, self).__init__(16) # has to be hard coded :(
        filename = os.path.join(os.path.dirname(__file__), 'bloodTypeVectors.csv')
        feasible_sets = np.genfromtxt(filename,skip_header=1, delimiter=',').transpose()
        self.matcher = GurobiMatcher(feasible_sets)
        self.arrival_daily_mean = np.array([
            32.3250498, 21.26307068, 9.998228937, 2.540044521, 14.54586926, 9.568116614, 4.499078324, 1.142988355, 4.531473682, 2.980754732, 1.401597571, 0.356075086, 0.548292189, 0.360660715, 0.169588318, 0.043083818])

        self.departure_daily_mean = np.array([14.39983629, 7.932098763, 4.343646604, 0.73473771, 6.479746736, 3.569345515, 1.954586798, 0.330622806, 2.01863507, 1.111957971, 0.608912295, 0.102998901, 0.24424766, 0.134542957, 0.073676221, 0.0124625])

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
            self.state[i] -= 1.0 * np.random.poisson(lam=self.departure_daily_mean[i])
            self.state[i] += 1.0 * np.random.poisson(lam=self.arrival_daily_mean[i])
            if self.state[i] < 0:
                self.state[i] = 0.0

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
