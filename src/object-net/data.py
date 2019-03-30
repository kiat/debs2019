"""
@author: samba
"""
import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans

# Visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plotly.offline import plot , iplot
import plotly.graph_objs as go

# Remove the outliers
def remove_outliers(file_path):
    # return the list of dataframes
    dataframe_lists = []
    outliers = []
    # Creating the dataframe and selecting the required columns
    for i in range(1):
        temp_out = pd.DataFrame()
        df = pd.read_csv(file_path, usecols=[1,2,3,4], skiprows=i*72000, nrows = 72000, names=["lz","X","Y","Z"])
        df['radiusSquare'] = df['X']*df['X']+df['Y']*df['Y']+df['Z']*df['Z']
        df['radius'] = np.sqrt(df['radiusSquare']).round(1)
        df['freq'] = df.groupby(['lz','radius'])['radius'].transform('count')
        for j in range(64):
            maxfreq = df[(df['lz']==j) & (df['radius']!=0)]['freq'].max()
            while maxfreq>75:
                temp_out =  temp_out.append(df.loc[(df['lz']==j) & (df['freq']==maxfreq)],ignore_index=True)
                df.drop(df[(df['lz']==j) & (df['freq']==maxfreq)].index, inplace=True)
                maxfreq = df[(df['lz']==j) & (df['radius']!=0)]['freq'].max()
            temp_out =  temp_out.append(df.loc[(df['lz']==j) & (df['radius']==0)],ignore_index=True)
            df.drop(df[(df['lz']==j) & (df['radius']==0)].index, inplace=True)
        outliers.append(temp_out.iloc[:,0:4])
        dataframe_lists.append(df.iloc[:,0:4])

    return outliers, dataframe_lists

# cluster the data points and get the results
def cluster(dataframes):
    points_with_labels = []
    for i in dataframes:
        kmeans_model = KMeans(n_clusters=3, random_state=1).fit(i)
        i['labels'] = kmeans_model.labels_
        points_with_labels.append(i)
    
    return points_with_labels

# Return the object data points
def object_points(dataframes):
    return_object_points = []
    for i in dataframes:
        i = i[i['labels']==i.labels.mode()[0]]
        return_object_points.append(i.iloc[:,:-1])
    
    return return_object_points

# Visualizing the data points
def visualize(data, type_of_vis = 1):
    if type(data) is pd.DataFrame:
        visualize_base(data,type_of_vis)
    else:
        count = 0
        for i in data:
            count+=1
            visualize_base(i, type_of_vis,count)

# Visualize each data frame
def visualize_base(data, type_of_vis, count=1):
    x = tuple(data['X'].tolist())
    y = tuple(data['Y'].tolist())
    z = tuple(data['Z'].tolist())
    if type_of_vis == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z,c='r',marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    
    else:
        if 'labels' in data:
            labels = data['labels'].tolist()
            trace = go.Scatter3d(x=x,y=y,z=z, mode='markers', marker=dict(color=labels,size=2,opacity=0.6))
        else:
            trace = go.Scatter3d(x=x,y=z,z=y, mode='markers', marker=dict(color='red',size=2,opacity=0.6))
            layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-100,100],),
                    yaxis = dict(
                        nticks=4, range = [-50,100],),
                    zaxis = dict(
                        nticks=4, range = [-100,100],),)
                  )
        data=[trace]
        fig = go.Figure(data=data, layout=layout)
        plot(fig,filename='../visuals/{}_{}.html'.format('vis',count), auto_open=False, show_link=False)