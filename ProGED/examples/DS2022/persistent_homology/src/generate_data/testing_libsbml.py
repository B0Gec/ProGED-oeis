
import pandas
import sbmltoodepy
import yaml2sbml

path_out = "D:\\Experiments\\MLJ23\\data\\sbml_models\\"

## from yaml to sbml
yaml_path_in = "C:\\Users\\NinaO\\PycharmProjects\\MLJ23\\Lib\\site-packages\\yaml2sbml\\doc\\examples\\Lotka_Volterra\\Lotka_Volterra_python\\"
yaml_filename = "Lotka_Volterra_basic.yml"
yaml2sbml.validate_yaml(yaml_dir=yaml_path_in + yaml_filename)

sbml_output_file = 'Lotka_Volterra_basic.xml'
yaml2sbml.yaml2sbml(yaml_path_in + yaml_filename, path_out + sbml_output_file)

# from sbml to python func
path_in = "C:\\Users\\NinaO\\PycharmProjects\\MLJ23\\Lib\\site-packages\\sbmltoodepy\\sbml_files\\"
model_name = "Lotka_Volterra_basic"
sbmltoodepy.ParseAndCreateModel(path_out + model_name + ".xml",
                                outputFilePath=path_out + model_name + ".py",
                                className="ModelName")

from data.sbml_models.Lotka_Volterra_basic import ModelName
modelInstance = ModelName()

modelInstance.RunSimulation(100)

modelInstance.time = 0

import numpy as np
times = np.zeros(101)
times[0] = modelInstance.time
concentrations = np.zeros((2, 101))
concentrations[0, 0] = modelInstance.s['x_1'].concentration
concentrations[1, 0] = modelInstance.s['x_2'].concentration

timeinterval = 1
for i in range(100):
	modelInstance.RunSimulation(timeinterval)
	times[i+1] = modelInstance.time
	concentrations[0, i+1] = modelInstance.s['x_1'].concentration
	concentrations[1, i+1] = modelInstance.s['x_2'].concentration

import matplotlib.pyplot as plt
plt.figure()
plt.plot(times, concentrations[0, :])
plt.plot(times, concentrations[1, :])
