import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

# note to self: matchers take in a set of counts
# output should be a coefficient for each of the valid sets
# the user of the matcher should sum these up, since they
# may want to do some randomization to simulate match failures


class BinaryMatcher:
    def __init__(self, valid_sets):
        # Valid sets should be a (dense?) numpy matrix.
        # we think in terms of column vectors
        self.valid_sets = valid_sets

    def match(self, state):
        raise NotImplementedError


class WeightedMatcher:
    def __init__(self, valid_sets):
        # Valid sets should be a (dense?) numpy matrix.
        # we think in terms of column vectors
        self.valid_sets = valid_sets

    def match(self, state, weights):
        raise NotImplementedError


def _obj_expr(model):
    return sum(model.a[i, j] * model.x[j] for j in model.J for i in model.I)


def _obj_expr_weights(model):
    return sum(model.a[i, j] * model.weights[i] * model.x[j] for j in model.J for i in model.I)

def _ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i, j] * model.x[j] for j in model.J) <= model.pool[i]


def arr_to_indexed_dict(arr, start_ind=1):
    result_dict = {}
    for i, val in enumerate(arr):
        result_dict[i+start_ind] = val
    return result_dict


def mat_to_indexed_dict(mat, start_ind=1):
    rowdim = mat.shape[0]
    coldim = mat.shape[1]
    result_dict = {}
    for i in range(rowdim):
        for j in range(coldim):
            result_dict[(i+start_ind, j+start_ind)] = mat[i, j]
    return result_dict


class PyomoMatcher(BinaryMatcher):
    def __init__(self, valid_sets, solver_name='glpk'):
        super(PyomoMatcher, self).__init__(valid_sets)
        self.n_types = self.valid_sets.shape[0]
        self.n_sets = self.valid_sets.shape[1]

        # create abstract model
        self.abstract = pyo.AbstractModel()
        self.abstract.n_types = pyo.Param(within=pyo.NonNegativeIntegers)
        self.abstract.n_sets = pyo.Param(within=pyo.NonNegativeIntegers)
        self.abstract.I = pyo.RangeSet(1, self.abstract.n_types)
        self.abstract.J = pyo.RangeSet(1, self.abstract.n_sets)
        self.abstract.a = pyo.Param(self.abstract.I, self.abstract.J)
        self.abstract.pool = pyo.Param(self.abstract.I, mutable=True)
        self.abstract.x = pyo.Var(
            self.abstract.J, domain=pyo.NonNegativeIntegers)
        self.abstract.OBJ = pyo.Objective(rule=_obj_expr, sense=pyo.maximize)
        self.abstract.AxbConstraint = pyo.Constraint(
            self.abstract.I, rule=_ax_constraint_rule)

        data_dict = {None: {'n_types': {None: self.n_types},
                     'n_sets': {None: self.n_sets},
                     'pool': arr_to_indexed_dict([10 for x in range(self.n_types)]),
                     'a': mat_to_indexed_dict(valid_sets)}}

        self.opt = pyo.SolverFactory(solver_name)
        self.instance = self.abstract.create_instance(data_dict)

    def match(self, state):
        for row in range(len(state)):
            self.instance.pool[row+1] = state[row]
        results = self.opt.solve(self.instance, tee=False)
        solns = [pyo.value(self.instance.x[i])
                 for i in range(1, self.n_sets+1)]

        return solns

class PyomoWeightedMatcher(WeightedMatcher):
    def __init__(self, valid_sets, solver_name='cplex_direct'):
        super(PyomoWeightedMatcher, self).__init__(valid_sets)
        self.n_types = self.valid_sets.shape[0]
        self.n_sets = self.valid_sets.shape[1]

        # create abstract model
        self.abstract = pyo.AbstractModel()
        self.abstract.n_types = pyo.Param(within=pyo.NonNegativeIntegers)
        self.abstract.n_sets = pyo.Param(within=pyo.NonNegativeIntegers)
        self.abstract.I = pyo.RangeSet(1, self.abstract.n_types)
        self.abstract.J = pyo.RangeSet(1, self.abstract.n_sets)
        self.abstract.a = pyo.Param(self.abstract.I, self.abstract.J)
        self.abstract.pool = pyo.Param(self.abstract.I, mutable=True)
        self.abstract.weights = pyo.Param(self.abstract.I, mutable=True)
        self.abstract.x = pyo.Var(
            self.abstract.J, domain=pyo.NonNegativeIntegers)
        self.abstract.OBJ = pyo.Objective(rule=_obj_expr_weights, sense=pyo.maximize)
        self.abstract.AxbConstraint = pyo.Constraint(
            self.abstract.I, rule=_ax_constraint_rule)

        data_dict = {None: {'n_types': {None: self.n_types},
                     'n_sets': {None: self.n_sets},
                     'pool': arr_to_indexed_dict([10 for x in range(self.n_types)]),
                     'weights': arr_to_indexed_dict([1 for y in range(self.n_types)]),
                     'a': mat_to_indexed_dict(valid_sets)}}

        self.opt = pyo.SolverFactory(solver_name)
        self.instance = self.abstract.create_instance(data_dict)

    def match(self, state, action):
        for row in range(len(state)):
            self.instance.pool[row+1] = state[row]
            self.instance.weights[row+1] = action[row]
        #self.instance.OBJ = pyo.Objective(expr=_obj_expr_weights_fn(action)(self.instance), sense=pyo.maximize)
        results = self.opt.solve(self.instance, tee=False)
        solns = [pyo.value(self.instance.x[i])
                 for i in range(1, self.n_sets+1)]

        return solns
