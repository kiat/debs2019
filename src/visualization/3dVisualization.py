#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:31:57 2019

@author: kia
"""



import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')




data= pd.read_csv('/home/kia/Collected-Datasets/DEBS2019/debs2019_initial_dataset/Atm/in.csv', sep=',', header=None, usecols=[2,3,4] , names=["X","Y", "Z"])
data = pd.DataFrame(data)


# Define a scnene number that we use to speare  data 
sceneNr = 3


# The use panda dataframe to slice it. 
# Each scene is 72k rows 
data1 = data.loc[ (1-sceneNr) * 72000 : sceneNr * 72000]

X=data1["X"]
Y=data1["Y"]
Z=data1["Z"]

ax.scatter(X, Y, Z)
 


plt.show()

