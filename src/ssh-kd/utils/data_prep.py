import math
import pandas as pd
import numpy as np
import pickle
from datetime import datetime


# Library for generating the random numbers
import random
from operator import itemgetter
import sys, os

sys.path.insert(0, os.path.abspath(".."))
from plugin.seg import remove_outliers, helper_object_points, max_min
from plugin.encode import input_nn
from plugin.load_model import object_names_func
from plugin.projections import prespective_project

# Fixed Outliers
def get_outliers(file_path):
    # Take the outliers
    outliers, others = helper_outliers("{}/Atm/in.csv".format(file_path))

    # get the min and max values for each outlier range
    for i in outliers:
        i["radiusSquare"] = i["X"] ** 2 + i["Y"] ** 2 + i["Z"] ** 2
        i["radius"] = np.sqrt(i["radiusSquare"]).round(1)
        i = i[i["radius"] > 0]
        i["max"] = i.groupby(["lz"])["radius"].transform("max")
        i["min"] = i.groupby(["lz"])["radius"].transform("min")
        i = i[["lz", "max", "min"]]
        i.drop_duplicates(subset=["lz", "max", "min"], inplace=True)

    # Save the values
    max_rad = []
    min_rad = []
    for j in range(64):
        max_rad.append(i[i["lz"] == j]["max"].tolist()[0])
        min_rad.append(i[i["lz"] == j]["min"].tolist()[0])
    total = (np.array(max_rad*1125),np.array(min_rad*1125))

    out_file = open("../data/outliers.pkl", 'wb')
    pickle.dump(total,out_file)
    out_file.close()


# Remove the outliers
def helper_outliers(file_path):
    # return the list of dataframes
    dataframe_lists = []
    outliers = []
    # Creating the dataframe and selecting the required columns
    for i in range(1):
        temp_out = pd.DataFrame()
        df = pd.read_csv(
            file_path,
            usecols=[1, 2, 3, 4],
            skiprows=i * 72000,
            nrows=72000,
            names=["lz", "X", "Y", "Z"],
        )
        df["radiusSquare"] = df["X"] * df["X"] + df["Y"] * df["Y"] + df["Z"] * df["Z"]
        df["radius"] = np.sqrt(df["radiusSquare"]).round(1)
        df["freq"] = df.groupby(["lz", "radius"])["radius"].transform("count")
        for j in range(64):
            maxfreq = df[(df["lz"] == j) & (df["radius"] != 0)]["freq"].max()
            while maxfreq > 75:
                temp_out = temp_out.append(
                    df.loc[(df["lz"] == j) & (df["freq"] == maxfreq)], ignore_index=True
                )
                df.drop(
                    df[(df["lz"] == j) & (df["freq"] == maxfreq)].index, inplace=True
                )
                maxfreq = df[(df["lz"] == j) & (df["radius"] != 0)]["freq"].max()
            temp_out = temp_out.append(
                df.loc[(df["lz"] == j) & (df["radius"] == 0)], ignore_index=True
            )
            df.drop(df[(df["lz"] == j) & (df["radius"] == 0)].index, inplace=True)
        outliers.append(temp_out.iloc[:, 0:4])
        dataframe_lists.append(df.iloc[:, 0:4])

    return outliers, dataframe_lists


def read_file(file_path, num_of_scenes=50):
    """Reads in the 50 scenes in to the memory and then slices the data frame to achieve optimization
    Takes 1.72 sec to read the 50 scenes file into memory
    Slicing each scene takes 0.32 milli seconds on average"""
    giant_df = pd.read_csv(
        file_path, usecols=[1, 2, 3, 4], names=["laser_id", "X", "Y", "Z"]
    )
    num_of_scenes = len(giant_df) / 72000
    dataframes = []

    for i in range(int(num_of_scenes)):
        start = i * 72000
        end = start + 72000
        dataframes.append(giant_df.iloc[start:end, :])

    return dataframes


def prepare_and_save_input(
    object_choice,
    object_names,
    folder_path,
    num_linear_transformations,
    num_of_scenes,
    path_to_pkl,
    grid_size,
    num_of_clusters,
    img_length,
    img_height,
    save_here,
    proj = False,
    proj_type = None,
    list_of_angles=[30, 60, 90, 120, 150],
):
    """
        Function for preparing the input for the desired object with linear transformations
        and two different views.
        Extend to get the training data from different angles
    """
    # Creating the path for the object
    object_path = "{}/{}/in.csv".format(folder_path, object_names[object_choice])

    # read in the object of choice
    dataframes = read_file(object_path, num_of_scenes)

    # remove the outliers
    no_outliers = remove_outliers(dataframes, num_of_scenes, path_to_pkl)

    # get the object points
    new_objects = []

    for i in no_outliers:
        new_objects.append(helper_object_points(i, num_of_clusters))
    
    if proj and proj_type == 'perspective':
        new_pres_objects = []
        grid_size = 0.1
        for i in new_objects:
            x,y,z = np.array(i['X']),np.array(i['Y']), np.array(i['Z'])
            new_pres_objects.append(prespective_project(x,y,z, 4, 2))
            new_pres_objects.append(prespective_project(x,y,z, 4, 3))
        # scale the points
        x_max_x, x_min_x, y_max_x, y_min_x = max_min(new_pres_objects, img_length, img_height, 2)

        # get the input for each scene with 4 different linear transformations
        n = len(new_pres_objects)

        input_data = []
        output = []

        for i in range(n):

            # get the objects with various number of linear transformations
            for j in range(num_linear_transformations):
                # creating the input in the XY view
                input_data.append(
                    input_nn(
                        new_pres_objects[i],
                        [x_min_x[j], x_max_x[j]],
                        [y_min_x[j], y_max_x[j]],
                        grid_size,
                        img_length,
                        img_height,
                        2,
                    )
                )
                output.append(object_names[object_choice])
        
    else:
        # scale the points
        x_max_x, x_min_x, y_max_x, y_min_x = max_min(new_objects, img_length, img_height, 2)
        x_max_z, x_min_z, y_max_z, y_min_z = max_min(new_objects, img_length, img_height, 3)

        # get the input for each scene with 4 different linear transformations
        n = len(new_objects)

        input_data = []
        output = []

        for i in range(n):

            # get the objects with various number of linear transformations
            for j in range(num_linear_transformations):
                # creating the input in the XY view
                input_data.append(
                    input_nn(
                        new_objects[i],
                        [x_min_x[i], x_max_x[i]],
                        [y_min_x[i], y_max_x[i]],
                        grid_size,
                        img_length,
                        img_height,
                        2,
                    )
                )
                output.append(object_names[object_choice])

                # Creating the input for the ZY view
                input_data.append(
                    input_nn(
                        new_objects[i],
                        [x_min_z[i], x_max_z[i]],
                        [y_min_z[i], y_max_z[i]],
                        grid_size,
                        img_length,
                        img_height,
                        3,
                    )
                )
                output.append(object_names[object_choice])

            # Save the objects
    np.save(
        "{}/{}_input.npy".format(save_here, object_names[object_choice]), input_data
    )
    np.save("{}/{}_output.npy".format(save_here, object_names[object_choice]), output)


def train_test_split(object_id, object_names, target, train_percent=0.7):
    """
        Function that creates the train and test split
    """
    # Input file and output file names
    file_name = "{}/{}_input.npy".format(target, object_names[object_id])
    output_file = "{}/{}_output.npy".format(target, object_names[object_id])

    # loading the saved npy format data
    object_input = np.load(file_name)
    object_output = np.load(output_file)

    # Creating the random shuffle data
    c = list(range(len(object_input)))

    # use random.Random(4) for fixing the seed
    random.shuffle(c)

    num_of_train = int(round(train_percent * len(c)))
    train = c[:num_of_train]
    test = c[num_of_train:]

    # Collecting the items of this index into
    train_input = list(itemgetter(*train)(object_input))
    test_input = list(itemgetter(*test)(object_input))
    train_output = list(itemgetter(*train)(object_output))
    test_output = list(itemgetter(*test)(object_output))

    # Adding them to the previous lists
    return train_input, test_input, train_output, test_output


def train_validation(input_train, output_train, batch_size):
    # zipping the input and output
    zipped_data = list(zip(input_train, output_train))

    # Shuffle them
    random.shuffle(zipped_data)

    # select the elements equal to batch_size
    zipped_data1 = zipped_data[:batch_size]
    zipped_data2 = zipped_data[batch_size:]
    # unzip the elements
    to_return = list(zip(*zipped_data1))
    x_batch = np.array(to_return[0], dtype=np.float32)
    y_batch = np.array(to_return[1], dtype=np.float32)

    to_return = list(zip(*zipped_data2))
    val_x = np.array(to_return[0], dtype=np.float32)
    val_y = np.array(to_return[1], dtype=np.float32)

    # return the elements
    return x_batch, y_batch, val_x, val_y

def data_prep(save_here, proj = False, proj_type=None):
    # list of individual objects
    list_of_object_choice = list(range(29))
    list_of_object_choice.remove(22)

    # objects
    object_names = object_names_func()

    # Required variables
    num_linear_transformations = 10
    num_of_scenes = 50
    path_to_pkl = "../data/outliers.pkl"
    grid_size = 0.1
    num_clusters = 4
    img_length = 10
    img_height = 7

    # Folder names
    dataset_dir = "../../dataset/train"
    # Create the outliers
    get_outliers(dataset_dir)

    # Preparing the data
    for i in list_of_object_choice:
        prepare_and_save_input(
            i,
            object_names,
            dataset_dir,
            num_linear_transformations,
            num_of_scenes,
            path_to_pkl,
            grid_size,
            num_clusters,
            img_length,
            img_height,
            save_here,
            proj,
            proj_type
        )


if __name__ == "__main__":
    save_here = "../data/"
    proj = True
    proj_type = 'perspective'
    
    if proj:
        data_prep(save_here, proj, proj_type)

    else:
        data_prep(save_here)