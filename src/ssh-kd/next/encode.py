# packages
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def input_nn(object_points, x_range, y_range, z_range, grid_size, img_length, img_height):
    """Function to create the input to the cnn function, need to see the time and optimize it later."""

    # create an empty numpy array for the counts
    n_rows = int(img_height / grid_size)
    n_cols = int(img_length / grid_size)

    # Populate the input array
    gridx = np.linspace(x_range[0], x_range[1], n_cols + 1)
    gridy = np.linspace(y_range[0], y_range[1], n_rows + 1)
    gridz = np.linspace(z_range[0], z_range[1], n_cols + 1)

    # according to the view
    x = np.array(object_points["X"])
    z = np.array(object_points["Z"])
    y = np.array(object_points["Y"])

    # compute the bi-dimensional histogram of two data samples
    grid, _ = np.histogramdd((x, y,z), bins=[gridx, gridy, gridz])

    # Returning the input
    return np.rot90(grid)


def flat_input(input_arr):
    """
        Converting the input arr into flat array and return the flattened input array for training.
    """
    to_return = []
    # Reshape each numpy array in the list
    for i in input_arr:
        to_return.append(i.flatten())
    return np.array(to_return)


def encode_output(output_arr):
    """
        Encodes the output arr to one hot encoding and return the encoders and encoded values.
    """
    # Transforming the output to one hot encoding, buy using label encoder at first and then one hot encoder
    encoder = LabelEncoder()
    encoded_out_test = encoder.fit_transform(output_arr)

    # using one hot encoder
    encoder1 = OneHotEncoder()
    encoded_out_test = encoder1.fit_transform(encoded_out_test.reshape(-1, 1)).toarray()

    # returning the encoders and encoded values
    return encoded_out_test, encoder, encoder1
