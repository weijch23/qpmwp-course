############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 2
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# %reload_ext autoreload
# %autoreload 2




# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import qpsolvers

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import load_data_msci
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.constraints import Constraints
from optimization.quadratic_program import QuadraticProgram
from optimization.optimization import MeanVariance







# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 10
data = load_data_msci(path = '../data/', n = N)
data





# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix
# --------------------------------------------------------------------------

X = data['return_series']
scalefactor = 1  # could be set to 252 (trading days) for annualized returns


expected_return = ExpectedReturn(method='geometric', scalefactor=scalefactor)
expected_return.estimate(X=X, inplace=True)

covariance = Covariance(method='pearson')
covariance.estimate(X=X, inplace=True)





# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------

# Instantiate the class
constraints = Constraints(ids = X.columns.tolist())

# Add budget constraint
constraints.add_budget(rhs=1, sense='=')

# Add box constraints (i.e., lower and upper bounds)
constraints.add_box(lower=0, upper=0.2)

# Add linear constraints
G = pd.DataFrame(np.zeros((2, N)), columns=constraints.ids)
G.iloc[0, 0:5] = 1
G.iloc[1, 6:10] = 1
h = pd.Series([0.5, 0.5])
constraints.add_linear(G=G, sense='<=', rhs=h)


constraints.budget
constraints.box
constraints.linear




# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios - using class QuadraticProgram
# --------------------------------------------------------------------------


# Extract the constraints in the format required by the solver
GhAb = constraints.to_GhAb()
GhAb


risk_aversion = 1

qp = QuadraticProgram(
    P = covariance.matrix.to_numpy() * risk_aversion,
    q = expected_return.vector.to_numpy() * -1,
    G = GhAb['G'],
    h = GhAb['h'],
    A = GhAb['A'],
    b = GhAb['b'],
    lb = constraints.box['lower'].to_numpy(),
    ub = constraints.box['upper'].to_numpy(),
    # solver = 'gurobi'
    solver = 'cvxopt'
)


qp.problem_data

qp.is_feasible()

qp.solve()
qp.results.get('solution')

qp.objective_value()







# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios - using class MeanVariance
# --------------------------------------------------------------------------


mv = MeanVariance(
    covariance = covariance,
    expected_return = expected_return,
    constraints = constraints,
    risk_aversion = 1
)


mv.params



mv.set_objective(optimization_data = data)
mv.objective.coefficients

mv.solve()
mv.results



