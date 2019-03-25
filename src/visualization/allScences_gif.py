import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class All_Scenes_Gif:

    # Initializing the object
    def __init__(self, path):
        self.path = path
    # Removing the outliers
    def remove_outliers(self,no_of_scenes=50):
        objects = []
        for j in range(no_of_scenes):
            df = pd.read_csv(self.path, usecols=[1,2,3,4], skiprows=j*72000, nrows=72000, names=["lz","X","Y","Z"])
            df['radiusSquare'] = df['X']*df['X']+df['Y']*df['Y']+df['Z']*df['Z']
            df['radius'] = np.sqrt(df['radiusSquare']).round(1)
            df['freq'] = df.groupby(['lz','radius'])['radius'].transform('count')
            for i in range(64):
                maxfreq = df[(df['lz']==i) & (df['radius']!=0)]['freq'].max()
                while maxfreq>100:
                    df.drop(df[(df['lz']==i) & (df['freq']==maxfreq)].index,inplace=True)
                    maxfreq = df[(df['lz']==i) & (df['radius']!=0)]['freq'].max()
                df.drop(df[(df['lz']==i) & (df['radius']==0)].index,inplace=True)
            # Creating the tuples
            x = tuple(df['X'].tolist())
            y = tuple(df['Y'].tolist())
            z = tuple(df['Z'].tolist())
            objects.append((x,y,z))
        
        return objects

    # plot the two axes
    def plot_two_axes(self,object_points, object_name, x = 0, y = 1):
        count  = 0
        for i in object_points:
            plt.scatter(i[x],i[y])
            plt.title(object_name)
            plt.xlabel(x)
            plt.ylabel(y)
            if count<10:
                name = "0"+str(count)
            else:
                name = str(count)
            plt.savefig("../visuals/_"+name+"_"+object_name+".png")
            plt.clf()
            count+=1

    # Create the gif
    def create_gif(self,object_name,x=0,y=1):
        os.chdir("../visuals")
        os.system("convert -delay 50 -loop 0 *{}.png {}_{}_{}.gif".format(object_name,object_name,x,y))