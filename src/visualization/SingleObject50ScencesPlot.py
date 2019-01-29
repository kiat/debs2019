#!/usr/bin/python3

"""
@author: samba
"""
import pandas
import numpy as np
import sys
from plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go
from datetime import datetime

if __name__ == "__main__":
    # Checking the number of arguments
    if len(sys.argv)!=2:
        print("Usage: python3 pointCloudsFromXYZ.py <path_to_directory_of_csv file> or ./pointCloudsFromXYZ.py <path_to_directory_of_csv>",file=sys.stderr)
        exit(-1)
    
    # Path to the file
    directory_path = sys.argv[1]
    if directory_path[-1]=='/':
        directory_path = directory_path[:-1]
    
    file_path = directory_path+'/in.csv'
    
    '''
    Creating the visualizations for 50 objects all at a time
    '''
    giant_df = pandas.DataFrame(columns=['lz','X','Y','Z','radiusSquare','radius','freq'])
    # Reading in the data for one scene at a time
    for j in range(50):
        starting = datetime.now()
        df = pandas.read_csv(file_path,usecols=[1,2,3,4],skiprows=j*72000,nrows=72000,names=["lz","X","Y", "Z"])
        df['radiusSquare'] = df['X']*df['X']+df['Y']*df['Y']+df['Z']*df['Z']
        df['radius'] = np.sqrt(df['radiusSquare']).round(1)
        df['freq'] = df.groupby(['lz','radius'])['radius'].transform('count')
        for i in range(64):
            maxfreq = df[(df['lz']==i) & (df['radius']!=0)]['freq'].max()
            while maxfreq>100:
                df.drop(df[(df['lz']==i) & (df['freq']==maxfreq)].index,inplace=True)
                maxfreq = df[(df['lz']==i) & (df['radius']!=0)]['freq'].max()
            df.drop(df[(df['lz']==i) & (df['radius']==0)].index,inplace=True)
        print('{} scene time taken for removing noise = {}'.format(j,datetime.now()-starting))
        giant_df = giant_df.append(df)
    # Creating the tuples
    x = tuple(giant_df['X'].tolist())
    y = tuple(giant_df['Y'].tolist())
    z = tuple(giant_df['Z'].tolist())

    # Creating the 3d mesh
    trace = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(color='rgb(0, 0, 0)',size=2,opacity=0.6))
    data = [trace]
    
    # Remove the points which is not representing an object

    # plot the points on 3d plot and saving it.
    plot(data,filename=directory_path[directory_path.rfind('/')+1:]+'{}.html'.format(j),auto_open=False,show_link=False)