"""
@author: samba
"""
import pandas as pd
import numpy as np
import sys

# Remove the outliers
def remove_outliers(file_path,num_of_scene):
    # Creating the dataframe and selecting the required columns
    df = pd.read_csv(file_path, usecols=[1,2,3,4], skiprows=num_of_scene*72000, nrows = 72000, names=["lz","X","Y","Z"])
    df['radiusSquare'] = df['X']*df['X']+df['Y']*df['Y']+df['Z']*df['Z']
    df['radius'] = np.sqrt(df['radiusSquare']).round(1)
    df['freq'] = df.groupby(['lz','radius'])['radius'].transform('count')
    for i in range(64):
        maxfreq = df[(df['lz']==i) & (df['radius']!=0)]['freq'].max()
        while maxfreq>100:
            df.drop(df[(df['lz']==i) & (df['freq']==maxfreq)].index, inplace=True)
            maxfreq = df[(df['lz']==i) & (df['radius']!=0)]['freq'].max()
        df.drop(df[(df['lz']==i) & (df['radius']==0)].index, inplace=True)
    return df

# cluster the data points and get the results
def cluster():
    pass

# Return the object data points
def object_points():
    pass