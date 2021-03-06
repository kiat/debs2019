# Packages
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans

# ToDo: optimize this function by storing the below lists of max and min radius and by considering only up certain laser number. 
def remove_outliers(dataframes, number_of_scenes=1, path_to_pkl="../outliers.pkl"):
    """Takes 0.36 sec to remove the outliers in each scene,Function to remove outliers"""    
    object_points = []
    outliers = pd.read_pickle(path_to_pkl)
    max_rad = []
    min_rad = []
    for i in range(64):
        max_rad.append(outliers[outliers["lz"] == i]["max"].tolist()[0])
        min_rad.append(outliers[outliers["lz"] == i]["min"].tolist()[0])

    for i in range(number_of_scenes):
        df = dataframes[i]
        df["radius"] = df.X.pow(2).add(df.Y.pow(2).add(df.Z.pow(2))).pow(0.5).round(1)
        df.drop(df[df["radius"] == 0].index, inplace=True)
        temp_out = pd.DataFrame()
        for j in range(64):
            dummy_df = df[df["laser_id"] == j]
            bool_vec = ~(
                (dummy_df["radius"] <= max_rad[j]) & (dummy_df["radius"] >= min_rad[j])
            )
            temp_out = temp_out.append(dummy_df[bool_vec])
        object_points.append(temp_out)

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


# TODO: think to remove this function
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
        y_mean = i["Y"].mean()
        first, second = generate_random(img_height)

        # Appending the max and min values
        y_max.append(y_mean + first)
        y_min.append(y_mean - second)

        # if the view is XY calcualte for X
        if view == 2:

            # using the mean to calculate the y_max and y_min
            x_mean = i["X"].mean()
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


# TODO: think to remove this function
def generate_random(residual):
    """
    Helper function for max_min to generate random numbers
    """
    if residual < 0:
        return 0, 0
    first = random.randint(1, 20)
    sec = random.randint(1, 20)
    tot = float(first + sec)
    return (first * residual / tot, sec * residual / tot)
