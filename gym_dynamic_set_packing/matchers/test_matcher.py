from . import GurobiMatcher
import numpy as np

def test_simple_matcher():
    n_types = 5
    n_sets = 3
    feasible_sets = np.zeros((n_types, n_sets))
    feasible_sets[0:2, 0] = 1.0
    feasible_sets[1:4, 1] = 1.0
    feasible_sets[3:5, 2] = 1.0

    pool = 10*np.ones(5)
    pool[0] = 8
    pool[4] = 8
    matcher = GurobiMatcher(feasible_sets)
    result = matcher.match(pool)
    assert result == [8.0, 2.0, 8.0]

