import warnings
import sys
import os
warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp
from pysindy.utils import linear_damped_SHO

import pysindy as ps

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

dt = 0.01
t_train = np.arange(0, 25, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [2, 0]
x_train = solve_ivp(linear_damped_SHO, t_train_span,
                    x0_train, t_eval=t_train, **integrator_keywords).y.T

x, y = x_train[:, 0], x_train[:, 1]
t = t_train
print(x.shape, x)

dxdt = np.gradient(x, dt)
dydt = np.gradient(y, dt)
dot_x = np.hstack((dxdt.reshape(-1,1), dydt.reshape(-1,1)))
print(x.shape, y.shape, t.shape, dot_x.shape)

# Fit the model
poly_order = 8
threshold = 10.05

sys.stderr = open(os.devnull, "w")  # silence stderr

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
    feature_names=['x', 'y']
)

sys.stderr = sys.__stderr__  # unsilence stderr

model.fit(x_train, t=dt, x_dot=dot_x)
model.print()


