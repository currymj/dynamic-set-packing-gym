import gurobipy as gr
import numpy as np

# note to self: matchers take in a set of counts
# output should be a coefficient for each of the valid sets
# the user of the matcher should sum these up, since they
# may want to do some randomization to simulate match failures

class Matcher:
    def __init__(self, valid_sets):
        #Valid sets should be a (dense?) numpy matrix.
        # we think in terms of column vectors
        self.valid_sets = valid_sets

    def match(self, state):
        raise NotImplementedError

class PyomoMatcher(Matcher):
    def __init__(self, valid_sets):
        super(PyomoMatcher, self).__init__(valid_sets)


    # implement match!
    # implement lp solver!
class GurobiMatcher(Matcher):
    def __init__(self, valid_sets, show_output=False):
        super(GurobiMatcher, self).__init__(valid_sets)
        self.show_output = show_output

    def match(self, state):
        "Assume state is a 1d numpy vector of counts per type"
        n_types = state.shape[0]
        n_sets = self.valid_sets.shape[1]
        
        # type vectors must be same size
        assert(n_types == self.valid_sets.shape[0])
        m = gr.Model("match")
        m.setParam('OutputFlag', self.show_output)
        x_vars = [m.addVar(vtype=gr.GRB.INTEGER, lb=0, name='x_{}'.format(i)) for i in range(n_sets)]
        row_sums = [gr.LinExpr(self.valid_sets[i,:], x_vars) for i in range(n_types)]
        for i, row_sum in enumerate(row_sums):
            m.addConstr(row_sum <= state[i])
        m.setObjective(gr.quicksum(row_sums), gr.GRB.MAXIMIZE)
        m.optimize()
        solns = [x.x for x in x_vars]
        return solns

class WeightedMatcher:

    def __init__(self):
        raise NotImplementedError

    def match(self, state, type_weights):
        raise NotImplementedError
