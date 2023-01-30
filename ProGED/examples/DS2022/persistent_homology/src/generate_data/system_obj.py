import numpy as np
import itertools

import sympy
from scipy.integrate import solve_ivp, odeint
# from extensisq import BS45, BS45_i
import ProGED as pg

class System():

    def __init__(self, sys_name, sys_vars, **kwargs):

        self.sys_name = sys_name
        self.sys_vars = sys_vars
        self.num_vars = len(sys_vars)
        self.orig_expr = kwargs.get('orig_expr', [])
        self.sym_structure = kwargs.get('sym_structure', [])
        self.sym_params = kwargs.get('sym_params', [])
        self.simpl_structure = kwargs.get('simpl_structure', [])
        self.simpl_params = kwargs.get('simpl_params', [])
        self.inits = kwargs.get('inits', np.random.rand(self.num_vars))
        self.bounds = kwargs.get('bounds', [-10, 10])
        self.benchmark = kwargs.get('benchmark', 'unknown')
        self.grammar_type = kwargs.get('grammar_type', 'unknown')
        self.data_column_names = kwargs.get('data_column_names', sys_vars)
        self.sys_func = kwargs.get('sys_func', [])

    def simulate(self, iinit, initial_time=0, sim_step=0.01, sim_time=50, rtol=1e-12, atol=1e-12, method="LSODA"):

        sys_func = simulate_check_sys_func(self)
        init = self.get_init(iinit)
        t = np.arange(initial_time, sim_time, sim_step)
        X = odeint(sys_func, init, t, rtol=rtol, atol=atol, tfirst=True)
        #X = solve_ivp(sys_func, [t[0], t[-1]], init, t_eval=t, rtol=rtol, atol=atol, method=method)
        # return np.column_stack([t.reshape((len(t), 1)), X.y.T])
        return np.column_stack([t.reshape((len(t), 1)), X])

    def get_obs(self, set_obs):
        if set_obs == "full":
            return self.get_obs_full()
        elif set_obs == "part":
            return self.get_obs_part()
        elif set_obs == "all":
            return self.get_obs_full() + self.get_obs_part()
        else:
            Warning("Such 'set_obs' does not exist. Returning 'set_obs==all'.")
            return self.get_obs_full() + self.get_obs_part()

    def get_obs_part(self):
        combs = list(itertools.combinations(self.data_column_names, len(self.data_column_names)-1))
        return [list(combs[i]) for i in range(len(combs))]

    def get_obs_full(self):
        return [self.data_column_names]

    def get_init(self, iinit):
        if type(iinit) == int:
            return self.inits[iinit, :]
        elif isinstance(iinit, (np.ndarray, np.generic)) and np.ndim(iinit) == 1:
            return iinit
        else:
            Warning("Initial values not specified properly. Returning the first within the object.inits list.")
            return self.inits[0, :]

def custom_func(t, x, sys_func):
    return [sys_func[i](*x) for i in range(len(sys_func))]

def custom_func_with_time(t, x, sys_func):
    return [sys_func[i](*x, t) for i in range(len(sys_func))]

def simulate_check_sys_func(self):
    if self.sys_func == []:
        systemBox = pg.ModelBox()
        systemBox.add_system(self.sym_structure, symbols={"x": self.sys_vars, "const": "C"})
        self.sys_func = systemBox[0].lambdify(params=self.sym_params, list=True)

    if sympy.Symbol("t") in self.sys_vars:
        sys_func = lambda t, x: custom_func_with_time(t, x, self.sys_func)
    else:
        sys_func = lambda t, x: custom_func(t, x, self.sys_func)

    return sys_func

