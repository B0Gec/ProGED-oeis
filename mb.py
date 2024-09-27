import os
from cocoa_location import cocoa_location

print(2)
print(os.getcwd())
filename_loc = os.getcwd() + '/julia/mb/'
filename = 'runable.cocoa5'
filefull = filename_loc + filename
os.chdir(filename_loc + filename)
print(os.chdir(os.getcwd() + '/julia/mb'))
print(os.chdir(os.getcwd() + '/julia/mb'))
os.system('ls')
print(3)
