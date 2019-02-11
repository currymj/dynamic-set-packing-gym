import gurobipy as gr
import numpy as np

class Matcher:
    def __init__(self):
        raise NotImplementedError

    def match(self, state):
        raise NotImplementedError

class WeightedMatcher:

    def __init__(self):
        raise NotImplementedError

    def match(self, state, type_weights):
        raise NotImplementedError
