#!/usr/bin/python3

"""
@author: samba
"""
import pandas
import sys
from plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go

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
    Creating the visualizations for 50 objects one at a time
    '''
    # Reading in the data for one scene at a time
    for i in range(2):
        df = pandas.read_csv(file_path,usecols=[2,3,4],skiprows=i*72000,nrows=72000,names=["X","Y", "Z"])
        
        # Creating the tuples
        x = tuple(df['X'].tolist())
        y = tuple(df['Y'].tolist())
        z = tuple(df['Z'].tolist())

        # Creating the 3d mesh
        trace = go.Mesh3d(x=x,y=y,z=z,color='grey',opacity=0.10)
        trace2 = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(color='rgb(0, 0, 0)',size=2,opacity=0.6))

        # plot the points on 3d plot and saving it.
        plot([trace,trace2],filename=directory_path[directory_path.rfind('/')+1:]+'{}.html'.format(i),auto_open=False,show_link=False)