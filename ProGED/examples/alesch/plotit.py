import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.interactive(False)

# true_data = np.array(pd.read_csv("C:\\Users\\ninao\\Downloads\\data.csv"))
# true_data = np.array(pd.read_csv("C:\\Users\\ninao\\Downloads\\data.csv"))
true_data = np.array(pd.read_csv('ales.csv'))
print(true_data[0:3,])
# 1/0

x = true_data[:, 0]
y = true_data[:, 1]
y_model1 = -0.000844809488168609*np.exp(5.86846085421262*x**3) + 0.449053901813809 - 0.458382100647211*np.exp(-1.60983823607706*x)
y_model2 = -0.174866381084018*x**2 + 0.581674288856877*x - 1.49375914626493e-5*np.exp(9.99947976270835*x)
y_model3 = 0.456650929924767*x - 0.0938455286879561*np.exp(1.66709723195817*x**5) + 0.110281937786071
y_model4_ext = -0.326042380413172*x**4 + 0.538705946890926*x - 8.19647260665275e-6*np.exp(9.99933952554213*x**6)
y_model5_focus = -0.324103923928674*x**4 + 0.538199288765982*x - 8.46196500099237e-6*np.exp(9.99984779315027*x**6)
# =  -0.320140550000333*x**4 + 0.537102454782314*x - 8.97446873149301e-6*exp(9.99990181314469*x**6), p = 1.0, parse trees = 1, valid = True, error = 8.623881276450192e-06, time = 220.36685466766357
y_model6_wtf = -0.320140550000333*x**4 + 0.537102454782314*x - 8.97446873149301e-6*np.exp(9.99990181314469*x**6)
y_model7 = -0.330405806823603*x**4 + 0.539989390536756*x - 3.12207505015483e-6*np.exp(10.965886402435*x**6)
y_model8 = - 0.329676405726662 * x ** 4 + 0.53971439899236 * x - 1.20042040174795e-6 * np.exp( 11.9999918106918 * x ** 6) # trees = 1, valid = True, error = 6.768838393765446e-06object
y_model9 =  -0.331423851484103*x**4 + 0.540261694120571*x - 2.05891210350665e-6*np.exp(11.3924602151874*x**6)  # , p = 1.0, parse trees = 1, valid = True, error = 1.721228811552053e-05, time = 2035.5011780261993



res_zero = np.mean((y - y) ** 2)
for y_model in [y, y_model1, y_model2, y_model3, y_model4_ext, y_model5_focus,
                y_model6_wtf,
                y_model7,
                y_model8,
                y_model9,
                ]:
    res = np.mean((y - y_model) ** 2)
    print(res)


# 1/0

plt.figure()
plt.plot(x, y, label="y true")
# plt.plot(x, y_model1, label="y model 1")
# plt.plot(x, y_model2, label="y model 2")
# plt.plot(x, y_model3, label="y model 3")
plt.plot(x, y_model8, label="y model 8 focus")
# plt.plot(x, y_model9, label="y model 9 focus")
plt.plot(x, y_model4_ext, label="y model 4 ext")
# plt.plot(x, y_model5_focus, label="y model 5 focus")
plt.plot(x, y_model7, label="y model 7 focus")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()



#
# import pandas as pd
# import numpy as np
# # from matplotlib import
# import argparse
#
# csv = pd.read_csv('ales.csv')
# cs = np.array(csv)
# # cso = np.hstack((cs, np.zeros((cs.shape[0], 1))))
#
# from ProGED.equation_discoverer import EqDisco
#
# # if __name__ == "__main__":
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.style.use('_mpl-gallery')
#
# # make data
# x = np.linspace(0, 10, 100)
# y = 4 + 2 * np.sin(2 * x)
# x = cs[:, 0]
# y = cs[:, 1]
#
# # plot
# fig, ax = plt.subplots()
#
# ax.plot(x, y, linewidth=0.1)
# # ax.plot(x, y)
#
# # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
# #        ylim=(0, 8), yticks=np.arange(1, 8))
#
#

from ProGED import ModelBox
from ProGED.parameter_estimation import fit_models

models = ModelBox()
exprs = ["C*x",
         "C*x + C",
         "C*x + C*exp(C*x)",
         ]
exprs = [
 "C*x**4 + C*x + C*exp(C*x**6)"
]

symbols = {"x": ['x'], "const": "C", "start": "S"}


for expr in exprs:
    models.add_model(expr, symbols)

estimation_settings = {'target_variable_index': -1,
                       "objective_settings": {"focus": (1, 0.2), },
                       'optimizer_settings': {
                           "lower_upper_bounds": (-5, 12),
                           "max_iter": 18000,
                           "pop_size": 150,
                       }
                       }
print(estimation_settings)

models = fit_models(models,
                    data=true_data,
                    task_type="algebraic",
                    pool_map=map,
                    estimation_settings=estimation_settings,
                    )



# fit_models()
# print(ED.models.retrieve_best_models(10000))
# print(ED.models)
# print(models.retrieve_best_models(-1))
print(models.retrieve_best_models(100))



# for model in models:
#     testY = model.evaluate(X, *params)
#     res = np.mean((Y - testY) ** 2)
#
# plt.show()
# plt.clf()
#
#
# #
# # 0.274 - 1.995/(32.78 - 15.22 sqrt(-x_0 + 6.647-7exp(14.569*x_0) + 0.775))
# # (0.03x0 - 0.063)(−11.104x0+(0.931x0−0.05)tan(3.22x0−1.74)+0.46)
