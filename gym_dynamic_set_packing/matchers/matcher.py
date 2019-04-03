import gurobipy as gr
import numpy as np

# note to self: matchers take in a set of counts
# output should be a coefficient for each of the valid sets
# the user of the matcher should sum these up, since they
# may want to do some randomization to simulate match failures

class BinaryMatcher:
    def __init__(self, valid_sets):
        #Valid sets should be a (dense?) numpy matrix.
        # we think in terms of column vectors
        self.valid_sets = valid_sets

    def match(self, state):
        raise NotImplementedError

class WeightedMatcher:
    def __init__(self, valid_sets):
        #Valid sets should be a (dense?) numpy matrix.
        # we think in terms of column vectors
        self.valid_sets = valid_sets

    def match(self, state, weights):
        raise NotImplementedError

class GurobiMatcher(BinaryMatcher):
    def __init__(self, valid_sets, show_output=False):
        super(GurobiMatcher, self).__init__(valid_sets)
        self.show_output = show_output
        self.n_types = self.valid_sets.shape[0]
        self.n_sets = self.valid_sets.shape[1]
        self.m = gr.Model("match")
        self.m.setParam('OutputFlag', self.show_output)
        self.x_vars = [self.m.addVar(vtype=gr.GRB.INTEGER, lb=0, name='x_{}'.format(i)) for i in range(self.n_sets)]
        self.row_sums = [gr.LinExpr(self.valid_sets[i,:], self.x_vars) for i in range(self.n_types)]
        self.m.setObjective(gr.quicksum(self.row_sums), gr.GRB.MAXIMIZE)
        for i, row_sum in enumerate(self.row_sums):
            self.m.addConstr(row_sum <= 0, name='rowsum_{}'.format(i))
            
        self.m.update()

    def match(self, state):
        "Assume state is a 1d numpy vector of counts per type"
        for i in range(len(self.row_sums)):
            constr = self.m.getConstrByName('rowsum_{}'.format(i))
            constr.setAttr(gr.GRB.Attr.RHS, state[i])
        
        self.m.optimize()
        solns = [x.x for x in self.x_vars]
        return solns

class GurobiWeightedMatcher(WeightedMatcher):
    def __init__(self, valid_sets, show_output=False):
        super(GurobiMatcher, self).__init__(valid_sets)
        self.show_output = show_output
        self.n_types = self.valid_sets.shape[0]
        self.n_sets = self.valid_sets.shape[1]
        self.m = gr.Model("match")
        self.m.setParam('OutputFlag', self.show_output)
        self.x_vars = [self.m.addVar(vtype=gr.GRB.INTEGER, lb=0, name='x_{}'.format(i)) for i in range(self.n_sets)]
        self.row_sums = [gr.LinExpr(self.valid_sets[i,:], self.x_vars) for i in range(self.n_types)]
        self.type_weights = np.ones(self.n_types)
        self.m.setObjective(gr.quicksum(self.row_sums), gr.GRB.MAXIMIZE)
        for i, row_sum in enumerate(self.row_sums):
            self.m.addConstr(row_sum <= 0, name='rowsum_{}'.format(i))
        self.m.update()

    def match(self, state, action):
        "Assume state is a 1d numpy vector of counts per type"
        for i in range(len(self.row_sums)):
            constr = self.m.getConstrByName('rowsum_{}'.format(i))
            constr.setAttr(gr.GRB.Attr.RHS, state[i])
        
        self.m.setObjective(gr.quicksum(row_sum[i] * action[i] for i in range(len(self.n_types))), gr.GRB.MAXIMIZE)
        self.m.optimize()
        solns = [x.x for x in self.x_vars]
        return solns

