# Packages
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN, MeanShift

def remove_outliers(dataframes, number_of_scenes=1, path_to_pkl="data/outliers.pkl"):
    """Takes 0.18 sec to remove the outliers in each scene,Function to remove outliers"""
    pd.options.mode.chained_assignment = None
    object_points = []
    outliers = pd.read_pickle(path_to_pkl)
    max_rad = outliers[0]
    min_rad = outliers[1]

    for i in range(number_of_scenes):

        df = dataframes[i]
        df.drop(columns=['laser_id'],inplace=True)
        df = df.to_numpy()
        rad = np.round(np.sqrt(df[:,0]**2+df[:,1]**2+df[:,2]**2),1)
        df = np.hstack([df,rad.reshape(-1,1)])
        bool_vec = (rad <= max_rad) & (rad >= min_rad)
        df = df[~bool_vec]
        # df = df[(df[:,1]<=10)]
        df = df[~(df[:,3]==0)]
        # bool_vec1 = (df[:,0]==0) & (df[:,2]==0)
        # df = df[~bool_vec1]
        object_points.append(df)

    return object_points


def helper_object_points(object_points, num_clusters):
    """Helper function to just return the object points by removing the """

    # cluster the points and get the labels
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1).fit(object_points)
    object_points["labels"] = kmeans_model.labels_

    # use the labels to slice the points and return the object points
    object_points = object_points[
        object_points["labels"] == object_points.labels.mode()[0]
    ]
    return object_points


def max_min(obj_data, img_length, img_height, view):
    """
    This function fits the points into the constant scale by calculating the x_max, x_min and y_max and y_min
    This works only for XY view or ZY view
    """
    # Lists to return which will help in plotting and collecting the data with in this range for the CNN
    x_max = []
    x_min = []
    y_max = []
    y_min = []

    ## going through each scene
    for i in obj_data:

        # using the mean to calculate the y_max and y_min
        y_mean = i[1].mean()
        first, second = generate_random(img_height)

        # Appending the max and min values
        y_max.append(y_mean + first)
        y_min.append(y_mean - second)

        # if the view is XY calcualte for X
        if view == 2:

            # using the mean to calculate the y_max and y_min
            x_mean = i[0].mean()
            first, second = generate_random(img_length)

            # Appending the max and min values
            x_max.append(x_mean + first)
            x_min.append(x_mean - second)

        # if the view is for ZY calcuate for Z
        elif view == 3:

            # using the mean to calculate the y_max and y_min
            z_mean = i["Z"].mean()
            first, second = generate_random(img_length)

            # Appending the max and min values
            x_max.append(z_mean + first)
            x_min.append(z_mean - second)

    return x_max, x_min, y_max, y_min


def generate_random(residual):
    """
    Helper function for max_min to generate random numbers
    """
    if residual < 0:
        return 0, 0
    first = random.randint(1, 20)
    sec = random.randint(1, 20)
    tot = float(first + sec)
    return first * residual / tot, sec * residual / tot


# Using the DB Scan of scikit learn for segmenting the data, take in the dataframe and return the labeled_df
# This DB scan is sensitive to starting point
# Come up with another idea
def segmentation(data, db_scan=True):
    if db_scan:
        # eps=2, min_sample=20 acc = 541335754464439
        clustering = DBSCAN(eps=2, min_samples=20).fit(data)
        labels = clustering.labels_
    else:
        clustering = MeanShift(bandwidth=2, bin_seeding=True, cluster_all=False, min_bin_freq=19, n_jobs=None, seeds=None).fit(data)
        labels = clustering.labels_

    return labels


# Try to use the top view to reduce the computation
def prepare_data(data_frame):

    labels = segmentation(
        np.array(
            list(
                zip(
                    data_frame[:,0],
                    data_frame[:,1],
                    data_frame[:,2]
                )
            )
        ),
        True
    )
    
    data_frame = np.hstack([data_frame, np.array(labels).reshape(-1,1)])

    return data_frame


# Extract the points of the clusters
def list_of_objects(dataframe):

    num_of_objects = np.unique(dataframe[:,4]).shape[0]-1
    list_objects = []
    for i in range(num_of_objects):
        list_objects.append(dataframe[dataframe[:,4] == i])

    return list_objects
