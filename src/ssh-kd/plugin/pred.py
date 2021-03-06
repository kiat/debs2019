import numpy as np
from plugin.seg import (
    remove_outliers,
    helper_object_points,
    max_min,
    prepare_data,
    list_of_objects,
)

from plugin.kde_seg import prep_obj_data
from plugin.encode import input_nn, flat_input
from plugin.projections import prespective_project

# Function for pred the input
def pred_scene(test_input, session, y_pred_cls, x):
    test_batch_size = 64
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
def predict(data_frame, session, img_length, img_height, y_pred_cls, x, proj=False, proj_type=None, segment_type=False):
    # prepare the data and segment it
    data_frame = remove_outliers([data_frame])[0]
    
    if segment_type:
        object_df = prep_obj_data(data_frame, False)
    else:
        segmented_df = prepare_data(data_frame)
        object_df = list_of_objects(segmented_df)

    dummy_input = []
    img_length = 10
    img_heigth = 7
    for j in object_df:
        
        if proj and proj_type=='perspective':

            # Numpy implementation
            X,y = prespective_project(j[:,0],j[:,1],j[:,2],4,3)
            x_max, x_min, y_max, y_min = max_min([[X,y]], img_length, img_height, 2)
        
            object_arr = input_nn(
                [X,y],
                [x_min[0], x_max[0]],
                [y_min[0], y_max[0]],
                0.1,
                img_length,
                img_height,
                2,
            )
            
            

        else:

            x_max, x_min, y_max, y_min = max_min([j[:,0],j[:,1]], img_length, img_height, 2)
            
            object_arr = input_nn(
                [j[:,0],j[:,1]],
                [x_min[0], x_max[0]],
                [y_min[0], y_max[0]],
                0.1,
                img_length,
                img_height,
                2,
            )
        object_arr = flat_input([object_arr]).tolist()[0]
        dummy_input.append(object_arr)
    return pred_scene(np.array(dummy_input), session, y_pred_cls, x)


# Take the predicted list and convert to dictionary of object counts for each scene
def convert_pred_to_dict(
    data_frame, session, object_names, img_length, img_height, y_pred_cls, x, proj, proj_type, segment_type
):
    output_pred = predict(data_frame, session, img_length, img_height, y_pred_cls, x, proj, proj_type, segment_type)
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
