"""
Note: Implemented with old implementation of remove_outliers.
"""
# packages
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("../ssh-kd"))
from plugin.seg import remove_outliers, helper_object_points
from utils.data_prep import read_file, get_outliers


# hidding the warnings
import warnings
warnings.filterwarnings('ignore')

# Clustering
from sklearn.cluster import DBSCAN


import matplotlib.pyplot as plt
def plot_g(x,y,xlim,ylim,filename,c=False,centroid=False):
    """ Plot the points using matplotlib.
        params: x = x points
                y = y points
                xlim = (x_max,x_min) xrange limits for plots
                ylim = (y_max,y_min) yrange limits for plots
                c = colors if avaliable else False
                centroid = centre point (lidar position)
        return: plot is plotted
    """
    fig = plt.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.gca()
    if c:
        plt.scatter(x, y, s=4, c=c)
    else:
        plt.scatter(x, y, s=4)
    if centroid:
        plt.scatter(0, 0, s=400, c="red")
    plt.grid()
    
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
#     plt.show()
    plt.savefig(filename)
    

def _cluster(X_):
    """ cluster using dbscan.
        params: X_ = list of x,y point
        return: lables
    """
    clustering = DBSCAN(eps=1, min_samples=8).fit(X_)
    labels = clustering.labels_
    return labels

def format_data(df):
    """ format data for cluster input.
        params: df = dataframe with transform _x,_y
        return: numpy array of x,y point
    """
    return np.array(list(zip(np.array(df["_x"]),np.array(df["_y"]))))

def normalize(df):
    """ Normalize the point using min_max normalizer.
        params: df = dataframe with transform _x,_y
        return: df with normalized _x,_y points
    """
    df['_x'] = (df['_x']-df['_x'].min())/(df['_x'].max()-df['_x'].min())
    df['_y'] = (df['_y']-df['_y'].min())/(df['_y'].max()-df['_y'].min())
    return df

def project(df, d, view):
    """ Project with prespective projections.
        formula: x' = x*d/z
                 y' = y*d/z
        params: df = dataframe with with x,y,z points
                d =  distance of prespective projection
                view = w.r.t X , w.r.t Z
        return: df with transformed _x,_y points in df
    """
    v= "X" if view == "X" else "Z"
    z= "X" if view == "Z" else "Z"
    df['_x'] = ((df[z]/df[v])*d)
    df['_y'] = ((df['Y']/df[v])*d)
    return df

def get_data(object_type,object,num_scenes):
    """ Load the data from dataset and apply ground removal and outlinear removal.
        params: object_type = train or test 
                object =  Name of the object
        return: dataframe of the data
    """
    dataset_dir = "../dataset/train"
    # Create the outliers
    get_outliers(dataset_dir)
    # Creating the path for the object
    object_path = "../{}/{}/in.csv".format("dataset/{}".format(object_type), object)
    # read in the object of choice
    dataframes = read_file(object_path, num_scenes)
    # remove the outliers
    no_outliers = remove_outliers(dataframes, num_scenes, '../ssh-kd/data/outliers.pkl')
    # get the object points
    return no_outliers


if __name__=="__main__":
    for dir_ in os.listdir('../dataset/train): 
        # Experimenting with train data'
        no_outliers = get_data("train",dir_, num_scenes=50)
        for each_scn in range(len(no_outliers)):
            # Get Each scene and work on it, scene selected 6
            scene_df = helper_object_points(no_outliers[each_scn], 4)
            # Apply projection
            proj_df = project(scene_df, 5,view="Z")
            # Apply normalization 
            proj_df = normalize(proj_df)
            # Plot the transformation
            try:
                os.makedirs('../projection/'+dir_+'/')
            except:
                pass
            plot_g(7*proj_df['_x'],3*proj_df['_y'],(-35,35),(-15,15),'../projection/'+dir_+'/'+str(each_scn)+'.png')