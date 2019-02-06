# note: this is just a scratch pad for me to familiarize myself with gurobi
# should be deleted eventually

from gurobipy import *
import numpy as np
import scipy.sparse

n_types = 5
n_sets = 3
feasible_sets = {}

feasible_sets = np.zeros((n_types, n_sets))

feasible_sets[0:2, 0] = 1.0
feasible_sets[1:4, 1] = 1.0
feasible_sets[3:5, 2] = 1.0

print(feasible_sets)
             
            

pool = 10*np.ones(5)
pool[0] = 8
pool[4] = 8
m = Model("match")

xs = [m.addVar(vtype=GRB.INTEGER, lb=0, name='x_{}'.format(i)) for i in range(n_sets)]

row_sums = [LinExpr(feasible_sets[i,:], xs) for i in range(n_types)]
for i, row_sum in enumerate(row_sums):
    m.addConstr(row_sum <= pool[i])

m.setObjective(quicksum(row_sums), GRB.MAXIMIZE)
m.optimize()

for x in xs:
    print(x.x)
