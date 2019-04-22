import numpy as np
from seg import (
    remove_outliers,
    helper_object_points,
    max_min,
    prepare_data,
    list_of_objects,
)
from encode import input_nn, flat_input

# Function for pred the input
def pred_scene(test_input, session):
    cls_pred = session.evaluate(test_input)
    return cls_pred


# Take the points of the cluster and fix the input using the above function
def predict(data_frame, session, img_length, img_height):
    # prepare the data and segment it
    data_frame = remove_outliers([data_frame])[0]
    segmented_df = prepare_data(data_frame)
    object_df = list_of_objects(segmented_df)
    dummy_input = []
    img_length = 16
    img_heigth = 16
    for j in object_df:
        x_max, x_min, y_max, y_min, z_max, z_min = max_min([j], img_length, img_height, 2)
        
        object_arr = input_nn(
            j,
            [x_min[0], x_max[0]],
            [y_min[0], y_max[0]],
            [z_min[0], z_max[0]],
            0.1,
            img_length,
            img_height,
            2
        )
        object_arr = flat_input([object_arr]).tolist()[0]
        dummy_input.append(object_arr)
    return pred_scene(np.array(dummy_input), session)


# Take the predicted list and convert to dictionary of object counts for each scene
def convert_pred_to_dict(
    data_frame, session, object_names, img_length, img_height
):
    output_pred = predict(data_frame, session, img_length, img_height)
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
