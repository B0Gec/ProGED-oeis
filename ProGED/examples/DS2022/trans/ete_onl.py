import pandas as pd
import os
system = 'myvdp'
data_ver = ""
method = 'dso'
data_ver = 'allonger'
leng = 2000 if data_ver == 'allonger' else 100
snr = 'inf'
init = 0
main_path = f"data{os.sep}{data_ver}{os.sep}"
path = f'{main_path}{system}{os.sep}'
file = f'{path}data_{system}_{data_ver}_len{leng}_snr{snr}_init{init}.csv'

# pd.read_csv("data_{system}_{pts}_len2000_snrinf_init0.csv")

csv = pd.read_csv(file)
dx = csv[['x', 'y', 'dx']]
dy = csv[['x', 'y', 'dy']]


# f = open(file, 'rb')
# print(f.read())
# f.close()
os.getcwd()
print(os.path.isdir(path))
print(os.path.isfile(file))
# os.listdir(path)
print(file)
print(path)

