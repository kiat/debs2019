# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 02:42:22 2019

@author: saeed
"""

import pandas
import sys
from plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go

if __name__ == "__main__":
     Checking the number of arguments
   if len(sys.argv)!=2:
      print("Usage: python3 pointCloudsFromXYZ.py <path_to_directory_of_csv file> or ./pointCloudsFromXYZ.py <path_to_directory_of_csv>",file=sys.stderr)
      exit(-1)
    
    # Path to the file
  
    directory_path =sys.argv[1]
    if directory_path[-1]=='/':
        directory_path = directory_path[:-1]         
    
    file_path = directory_path+'/in.csv'
      
    '''
    Creating the visualizations for 50 objects one at a time
    '''
    # Reading in the data for one scene at a time
    for i in range(50):        
        df = pandas.read_csv(file_path,skiprows=i*72000,nrows=72000,names=["POS","LaserID","X","Y", "Z"])
        
        objectpoints=pandas.DataFrame(columns = ["POS","LaserID","X","Y", "Z"])       
        #Find Radius for each laser
        for l in range(64):
            laserdata=df[df["LaserID"]==l]   
            # laserdata=laserdata[laserdata["Y"]>=-2.50]
            # Remove the points which is not representing an object
            #radius=(laserdata["X"].max()-laserdata["X"].min())/2
            radius=laserdata["X"].max();
            sdf=laserdata[(laserdata["X"]**2+laserdata["Z"]**2)<radius**2-0.5]
            objectpoints= objectpoints.append(sdf)           

        # Creating the 3d mesh
        trace = go.Scatter3d(x=objectpoints["X"],y=objectpoints["Y"],z=objectpoints["Z"],mode='markers',marker=dict(color='rgb(0, 0, 0)',size=2,opacity=0.6))
        data = [trace]       

        # plot the points on 3d plot and saving it.
        plot(data,filename=directory_path[directory_path.rfind('/')+1:]+'{}.html'.format(i),auto_open=False,show_link=False)
   
