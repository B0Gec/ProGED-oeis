from src.generate_data.system_obj import System
import numpy as np
import ProGED as pg
import sympy

n_inits = 10

sys_bacres = System(sys_name="bacres",
                    benchmark="strogatz",
                    sys_vars=["x", "y"],
                    orig_expr=["20 - x - ((x * y) / (1 + 0.5 * x**2))",
                               "10 - (x * y / (1 + 0.5 * x**2))"],
                    sym_structure=["C - (x*y / (C * x**2 + C)) - x",
                                    "C - (x*y / (C * x**2 + C))"],
                    sym_params=[[20, 0.5, 1], [10, 0.5, 1]],
                    inits=np.tile([5, 10], n_inits).reshape(n_inits, -1) + np.random.normal(1, 1, (n_inits, 2)),
                    bounds=[0, 20],
                    data_column_names=["x", "y"],
                    grammar_type=[],
                    )

sys_barmag = System(sys_name="barmag",
                    benchmark="strogatz",
                    sys_vars=["x", "y"],
                    orig_expr=["0.5 * sin(x-y) - sin(x)", "0.5 * sin(y-x) - sin(y)"],
                    sym_structure=["C * sin(x-y) - sin(x)", "C * sin(x-y) - sin(y)"],
                    sym_params=[[0.5], [-0.5]],
                    inits=np.tile([2*np.pi, 2*np.pi], n_inits).reshape(n_inits, -1) + np.random.uniform(0, 1, (n_inits, 2)),
                    bounds=[-5, 5],
                    data_column_names=["x", "y"],
                    grammar_type=[])

sys_glider = System(sys_name="glider",
                    benchmark="strogatz",
                    sys_vars=["x", "y"],
                    orig_expr=["-0.05 * x**2 - sin(y)", "x - (cos(y)/x)"],
                    sym_structure=["C * x**2 - sin(y)", "x - (cos(y)/x)"],
                    sym_params=[[-0.05], []],
                    inits=np.tile([5, 0], n_inits).reshape(n_inits, -1) + np.random.normal(1, 1, (n_inits, 2)),
                    bounds=[-5, 5],
                    data_column_names=["x", "y"],
                    grammar_type=[])

sys_lotka = System(sys_name="lv",
                   benchmark="strogatz",
                   sys_vars=["x", "y"],
                   orig_expr=["3*x - 2*x*y - x**2", "2*y - x*y - y**2"],
                   sym_structure=["C*x**2 + C*x*y + C*x", "C*y - x*y - y**2"],
                   sym_params=[[-1, -2, 3], [2]],
                   inits=np.random.randint(low=1, high=8, size=(n_inits, 2)),
                   bounds=[-5, 5],
                   data_column_names=["x", "y"],
                   grammar_type=[])

sys_predprey = System(sys_name="predprey",
                      benchmark="strogatz",
                      sys_vars=["x", "y"],
                      orig_expr=["x*(4 - x - y/(1+x))", "y*(x/(1+x) - 0.075*y)"],
                      sym_structure=["x * (C - x - y/(C + x))", "y * (C*y + x/(C + x))"],
                      sym_params=[[4, 1], [-0.075, 1]],
                      inits=np.tile([5, 10], n_inits).reshape(n_inits, -1) + np.random.normal(1, 1, (n_inits, 2)),
                      bounds=[-5, 5],
                      data_column_names=["x", "y"],
                      grammar_type=[])


sys_shearflow = System(sys_name="shearflow",
                       benchmark="strogatz",
                       sys_vars=["x", "y"],
                       orig_expr=["cot(y) * cos(x)", "(cos(y)**2 + 0.1*sin(y)**2) * sin(x)"],
                       sym_structure=["cos(x) * cot(y)", "(C*sin(y)**2 + cos(y)**2) * sin(x)"],
                       sym_params=[[], [0.1]],
                       inits=np.array([2*np.pi*np.random.rand(n_inits)-np.pi, np.pi*np.random.rand(n_inits)-np.pi/2]).reshape(n_inits, -1),
                       bounds=[-5, 5],
                       data_column_names=["x", "y"],
                       grammar_type=[])

sys_vdp = System(sys_name="vdp",
                 benchmark="strogatz",
                 sys_vars=["x", "y"],
                 orig_expr=["10 * (y - 1/3 * (x**3 - x))", "-0.1 * x"],
                 sym_structure=["C * (C * (x**3 - x) + y)", "C * x"],
                 sym_params=[[10, -1/3], [-1/10]],
                 simpl_structure=["C*x**3 + C*x + C*y", "C*x"],
                 simpl_params=[[-10/3, 10/3, 10], [-1/10]],
                 inits=np.random.rand(n_inits*2).reshape(n_inits, -1),
                 bounds=[-10, 10],
                 data_column_names=["x", "y"],
                 grammar_type=[],
                 )

sys_myvdp = System(sys_name="myvdp",
                   benchmark="custom",
                   sys_vars=["x", "y"],
                   orig_expr=["y", "-3*x + 2*y*(1 - x**2)"],
                   sym_structure=["y", "C*x + C*y*(1 - x**2)"],
                   sym_params=[[], [-3., 2.]],
                   simpl_structure=["y", "C*x**2*y + C*x + C*y"],
                   simpl_params=[[], [-2., -3., 2.]],
                   inits=np.random.uniform(low=-5, high=5, size=(n_inits*2,)).reshape(n_inits, -1),
                   bounds=[-5, 5],
                   data_column_names=["x", "y"],
                   grammar_type=[],
                   )

sys_stl = System(sys_name="stl",
                 benchmark="custom",
                 sys_vars=["x", "y"],
                 orig_expr=["-3*y + x*(1 - x**2 - y**2)", "3*x + y*(1 - x**2 - y**2)"],
                 sym_structure=["C*y + x*(C - x**2 - y**2)", "C*x + y*(C - x**2 - y**2)"],
                 sym_params=[[-3., 1.], [3., 1.]],
                 simpl_structure=["C*x + C*y - x**3 - x*y**2", "C*x + C*y - x**2*y - y**3"],
                 simpl_params=[[1., -3.], [3., 1.]],
                 inits=np.random.uniform(low=-5, high=5, size=(n_inits*2,)).reshape(n_inits, -1),
                 bounds=[-5, 5],
                 data_column_names=["x", "y"],
                 grammar_type=[],
                 )

sys_cphase = System(sys_name="cphase",
                    benchmark="custom",
                    sys_vars=["x", "y", "t"],
                    orig_expr=["C*sin(x) + C*sin(y) + C*sin(C*t) + C", "C*sin(x) + C*sin(y) + C"],
                    sym_structure=["C*sin(x) + C*sin(y) + C*sin(C*t) + C", "C*sin(x) + C*sin(y) + C"],
                    sym_params=[[0.8, 0.8, -0.5, 2*np.pi*0.0015, 2.], [0, 0.6, 4.53]],
                    inits=np.random.uniform(low=-5, high=5, size=(n_inits*2,)).reshape(n_inits, -1),
                    bounds=[-5, 5],
                    data_column_names=["x", "y"],
                    grammar_type=[],
                    )

sys_lorenz = System(sys_name="lorenz",
                    benchmark="custom",
                    sys_vars=["x", "y", "z"],
                    orig_expr=["10*(-x + y)", "28*x - x*z - y", "(-8/3)*z + x*y"],
                    sym_structure=["C*(-x + y)", "C*x - x*z - y", "C*z + x*y"],
                    sym_params=[[10.], [28.], [-8/3]],
                    simpl_structure=["C*x + C*y", "C*x - x*z - y", "C*z + x*y"],
                    simpl_params=[[10., 10.], [28.], [-8/3]],
                    inits=np.random.uniform(low=-5, high=5, size=(n_inits*3,)).reshape(n_inits, -1),
                    bounds=[-30, 30],
                    data_column_names=["x", "y", "z"],
                    grammar_type=[],
                    )

sys_vdpvdpUS = System(sys_name="vdpvdpUS",
                      benchmark="custom",
                      sys_vars=["x", "y", "u", "v"],
                      orig_expr=["y", "-3*x + 0.5*y*(1 - x**2)", "v", "-3*u + 0.5v*(1 - u**2) + 0.4*y**2"],
                      sym_structure=["y", "C*x + C*y*(1 - x**2)", "v", "C*u + C*v*(1 - u**2) + C*y**2"],
                      sym_params=[[], [-3., 0.5], [], [-3., 0.5, 0.4]],
                      simpl_structure=["y", "C*x**2*y + C*x + C*y", "v", "C*u**2*v + C*u + C*v + C*y**2"],
                      simpl_params= [[], [0.5, -3., 0.5], [], [0.5, -3., 0.5, 0.4]],
                      inits=np.random.uniform(low=-5, high=5, size=(n_inits*4,)).reshape(n_inits, -1),
                      bounds=[-5, 5],
                      data_column_names=["x", "y", "u", "v"],
                      grammar_type=[],
                      )

sys_stlvdpBL = System(sys_name="stlvdpBL",
                      benchmark="custom",
                      sys_vars=["x", "y", "u", "v"],
                      orig_expr=["-3*y + x*(1 - x**2 - y**2)", "3*x + y*(1 - x**2 - y**2) + 0.8*v", "v", "-3*u + 0.5*v*(1 - u**2) + 0.4*y"],
                      sym_structure=["C*y + x*(C - x**2 - y**2)", "C*x + y*(C - x**2 - y**2) + C*v", "v", "C*u + C*v*(1 - u**2) + C*y"],
                      sym_params=[[-3., 1.], [0.8, 3., 1.], [], [-3., 0.5, 0.4]], # freq w = 3; a = 1, eta = 0.5; couplings C1 = 0.8, C2 = 0.2
                      simpl_structure=["C*x + C*y - x**3 - x*y**2", "C*v + C*x + C*y - x**2*y - y**3", "v", "C*u**2*v + C*u + C*v + C*y"],
                      simpl_params=[[-3., 1.], [0.8, 3., 1.], [], [0.5, -3., 0.5, 0.4]],
                      inits=np.random.uniform(low=-5, high=5, size=(n_inits*4,)).reshape(n_inits, -1),
                      bounds=[-5, 5],
                      data_column_names=["x", "y", "u", "v"],
                      grammar_type=[],
                      )

strogatz = {
    sys_bacres.sys_name: sys_bacres,
    sys_barmag.sys_name: sys_barmag,
    sys_glider.sys_name: sys_glider,
    sys_lotka.sys_name: sys_lotka,
    sys_predprey.sys_name: sys_predprey,
    sys_shearflow.sys_name: sys_shearflow,
    sys_vdp.sys_name: sys_vdp,
}

mysystems = {
    sys_myvdp.sys_name: sys_myvdp,
    sys_stl.sys_name: sys_stl,
    sys_cphase.sys_name: sys_cphase,
    sys_lorenz.sys_name: sys_lorenz,
    sys_vdpvdpUS.sys_name: sys_vdpvdpUS,
    sys_stlvdpBL.sys_name: sys_stlvdpBL,
}
