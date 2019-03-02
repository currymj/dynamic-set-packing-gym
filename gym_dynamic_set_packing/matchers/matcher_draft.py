import numpy as np
import pyomo.environ as pyo

    #assert result == [8.0, 2.0, 8.0]

n_types = 5
n_sets = 3

model = pyo.AbstractModel()

model.n_types = pyo.Param(within=pyo.PositiveIntegers)
model.n_sets = pyo.Param(within=pyo.PositiveIntegers)
model.n_types_range = pyo.RangeSet(1, model.n_types)
model.n_sets_range = pyo.RangeSet(1, model.n_sets)
model.pool = pyo.Param(model.n_types_range, within=pyo.NonNegativeIntegers)
model.S = pyo.Param(model.n_types_range, model.n_sets_range, within=pyo.NonNegativeIntegers)
model.x = pyo.Var(model.n_sets_range, domain=pyo.NonNegativeIntegers)
def sx_product_row(model, row_ind):
    return sum(model.S[row_ind, j]*model.x[j] for j in range(pyo.value(model.n_sets)))

def sx_constraint_rule(model, row_ind):
    return (sx_product_row(model, row_ind) <= model.pool[row_ind])

def obj_function(model):
    return sum(sx_product_row(model, i) for i in pyo.value(model.n_types))

model.pool_constraint = pyo.Constraint(model.n_types_range, rule=sx_constraint_rule)

feasible_sets = np.zeros((n_types, n_sets))
feasible_sets[0:2, 0] = 1.0
feasible_sets[1:4, 1] = 1.0
feasible_sets[3:5, 2] = 1.0

pool = 10*np.ones(5)
pool[0] = 8
pool[4] = 8

def flatten_if_singleton(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup

def numpy_array_to_dict(arr):
    return {flatten_if_singleton(tup): elem for tup, elem in np.ndenumerate(arr)}


data_dict = {None:
        {
            'n_types': {None: n_types},
            'n_sets': {None: n_sets},
            'pool': numpy_array_to_dict(pool),
            'S': numpy_array_to_dict(feasible_sets)}}

instantiated = model.create_instance(data_dict)
