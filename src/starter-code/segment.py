import numpy as np
from sklearn.cluster import DBSCAN
from grouping import remove_outliers, helper_object_points, max_min_3d
from data import input_3dnn, flat_input


# Using the DB Scan of scikit learn for segmenting the data, take in the dataframe and return the labeled_df
# This DB scan is sensitive to starting point
# Come up with another idea
def segmentation(data):
    clustering = DBSCAN(eps=1, min_samples=16).fit(data)
    labels = clustering.labels_

    return labels


# Try to use the top view to reduce the computation
def prepare_data(data_frame):
    labels = segmentation(
        np.array(
            list(
                zip(
                    np.array(data_frame["X"]),
                    np.array(data_frame["Y"]),
                    np.array(data_frame["Z"]),
                )
            )
        )
    )
    data_frame["labels"] = labels

    return data_frame


# Extract the points of the clusters
def list_of_objects(dataframe):
    num_of_objects = dataframe["labels"].value_counts().index.shape[0] - 1
    list_objects = []
    for i in range(num_of_objects):
        list_objects.append(dataframe[dataframe["labels"] == i])

    return list_objects




# Take the points of the cluster and fix the input using the above function
def predict(data_frame,object_names, img_length, img_height):
    # prepare the data and segment it
    data_frame = remove_outliers([data_frame])[0]
    segmented_df = prepare_data(data_frame)
    object_df = list_of_objects(segmented_df)
    dummy_input = []
    img_length = 16
    img_height = 16
    for j in object_df:
        x_max, x_min, y_max, y_min, z_max, z_min = max_min_3d([j], img_length, img_height)
        object_arr = input_3dnn(
            j,
            [x_min[0], x_max[0]],
            [y_min[0], y_max[0]],
            [z_min[0], z_max[0]],
            1,
            img_length,
            img_height,
        )
        object_arr = flat_input([object_arr]).tolist()[0]
        dummy_input.append(object_arr)
    return np.array(dummy_input)


# Take the predicted list and convert to dictionary of object counts for each scene
def convert_pred_to_dict(output_pred, object_names):
    a = {}
    for j in output_pred:
        if j >= 22:
            if object_names[j + 1] in a:
                a[object_names[j + 1]] = a[object_names[j + 1]] + 1
            else:
                a[object_names[j + 1]] = 1
        else:
            if object_names[j] in a:
                a[object_names[j]] = a[object_names[j]] + 1
            else:
                a[object_names[j]] = 1

    return a
