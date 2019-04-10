import numpy as np
from sklearn.cluster import DBSCAN
from .grouping import remove_outliers, helper_object_points, max_min


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


# Get the labels of each scene
segmented_dataframe = []
error = 0
sum_ = 0
for i in range(0, len(test_outliers_dataframes)):
    temp = prepare_data(test_outliers_dataframes[i])
    a = len(temp["labels"].value_counts())
    b = sum(outfile_list[i].values())
    error += (a - b) * (a - b)
    sum_ += a - b
    print(
        "Detected no of objects {}, original no of objects {}, difference = {}".format(
            a, b, a - b
        )
    )
    segmented_dataframe.append(temp)

print(error)
print("sum {}".format(sum_))


# Extract the points of the clusters
def list_of_objects(dataframe):
    num_of_objects = dataframe["labels"].value_counts().index.shape[0] - 1
    list_objects = []
    for i in range(num_of_objects):
        list_objects.append(dataframe[dataframe["labels"] == i])

    return list_objects


# Function for pred the input
def pred_scene(test_input):
    num_obj = len(test_input)
    cls_pred = np.zeros(shape=num_obj, dtype=np.int)

    # Starting index
    i = 0

    while i < num_obj:
        j = min(i + test_batch_size, num_obj)

        # get the images
        images = test_input[i:j]

        # Calculate the predicted class using tensor flow
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict={x: images})

        i = j

    return cls_pred


# Take the points of the cluster and fix the input using the above function
output_pred = []

for i in segmented_dataframe:
    object_dataframes = list_of_objects(i)
    dummy_input = []
    for j in object_dataframes:
        x_max, x_min, y_max, y_min = max_min([j], img_length, img_height, 2)
        object_arr = input_nn(
            j,
            [x_min[0], x_max[0]],
            [y_min[0], y_max[0]],
            0.1,
            img_length,
            img_height,
            2,
        )
        object_arr = flat_input([object_arr]).tolist()[0]
        dummy_input.append(object_arr)
    # Using the saved_model predict the results
    output_pred.append(pred_scene(np.array(dummy_input)))


# Take the predicted list and convert to dictionary of object counts for each scene
def convert_pred_to_dict(output_pred):
    to_return = []
    for i in output_pred:
        a = {}
        for j in i:
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
        to_return.append(a)
    return to_return
