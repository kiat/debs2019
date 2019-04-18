def input_nn(object_points, x_range, y_range, grid_size, img_length, img_height, view):
    """Function to create the input to the cnn function, need to see the time and optimize it later"""
    # Need to check the time
    # start_time = datetime.now()

    # create an empty numpy array for the counts
    n_rows = int(img_height / grid_size)
    n_cols = int(img_length / grid_size)

    # input_list = np.zeros((n_rows, n_cols))

    # Populate the input array
    gridx = np.linspace(x_range[0], x_range[1], n_cols + 1)
    gridy = np.linspace(y_range[0], y_range[1], n_rows + 1)

    # according to the view
    if view == 2:
        x = np.array(object_points["X"])
    elif view == 3:
        x = np.array(object_points["Z"])

    y = np.array(object_points["Y"])

    grid, _, _ = np.histogram2d(x, y, bins=[gridx, gridy])

    #     print("Time taken = {}".format(datetime.now()-start_time))

    # Returning the input
    return np.rot90(grid)

def flat_input(input_arr):
    """
        Converting the input arr into flat array and return the flattened input array for training
    """
    # Reshape each numpy array in the list
    to_return = []
    for i in input_arr:
        to_return.append(i.flatten())
    return np.array(to_return)


def encode_output(output_arr):
    """
        Encodes the output arr to one hot encoding and return the encoders and encoded values
    """
    # Transforming the output to one hot encoding, buy using label encoder at first and then one hot encoder
    encoder = LabelEncoder()
    encoded_out_test = encoder.fit_transform(output_arr)

    # using one hot encoder
    encoder1 = OneHotEncoder()
    encoded_out_test = encoder1.fit_transform(encoded_out_test.reshape(-1, 1)).toarray()

    # returning the encoders and encoded values
    return encoded_out_test, encoder, encoder1
