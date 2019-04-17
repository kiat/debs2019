import math
import pandas as pd
import numpy as np
from datetime import datetime

# Library for generating the random numbers
import random
from operator import itemgetter
from functools import reduce

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from grouping import remove_outliers, helper_object_points, max_min

# Fixed Outliers
def get_outliers(file_path):
    # Take the outliers
    outliers , others = helper_outliers("{}/Atm/in.csv".format(file_path))

    # get the min and max values for each outlier range 
    for i in outliers:
        i['radiusSquare'] = i['X']**2+i['Y']**2+i['Z']**2
        i['radius'] = np.sqrt(i['radiusSquare']).round(1)
        i = i[i['radius']>0]
        i['max'] = i.groupby(['lz'])['radius'].transform('max')
        i['min'] = i.groupby(['lz'])['radius'].transform('min')
        i = i[['lz','max','min']]
        i.drop_duplicates(subset=['lz','max','min'],inplace=True)
    
    # Save the data frame
    i.to_pickle("../outliers.pkl")


# Remove the outliers
def helper_outliers(file_path):
    # return the list of dataframes
    dataframe_lists = []
    outliers = []
    # Creating the dataframe and selecting the required columns
    for i in range(1):
        temp_out = pd.DataFrame()
        df = pd.read_csv(file_path, usecols=[1,2,3,4], skiprows=i*72000, nrows = 72000, names=["lz","X","Y","Z"])
        df['radiusSquare'] = df['X']*df['X']+df['Y']*df['Y']+df['Z']*df['Z']
        df['radius'] = np.sqrt(df['radiusSquare']).round(1)
        df['freq'] = df.groupby(['lz','radius'])['radius'].transform('count')
        for j in range(64):
            maxfreq = df[(df['lz']==j) & (df['radius']!=0)]['freq'].max()
            while maxfreq>75:
                temp_out =  temp_out.append(df.loc[(df['lz']==j) & (df['freq']==maxfreq)],ignore_index=True)
                df.drop(df[(df['lz']==j) & (df['freq']==maxfreq)].index, inplace=True)
                maxfreq = df[(df['lz']==j) & (df['radius']!=0)]['freq'].max()
            temp_out =  temp_out.append(df.loc[(df['lz']==j) & (df['radius']==0)],ignore_index=True)
            df.drop(df[(df['lz']==j) & (df['radius']==0)].index, inplace=True)
        outliers.append(temp_out.iloc[:,0:4])
        dataframe_lists.append(df.iloc[:,0:4])

    return outliers, dataframe_lists

def read_file(file_path, num_of_scenes=50):
    """Reads in the 50 scenes in to the memory and then slices the data frame to achieve optimization
    Takes 1.72 sec to read the 50 scenes file into memory
    Slicing each scene takes 0.32 milli seconds on average"""
    giant_df = pd.read_csv(file_path, usecols=[1, 2, 3, 4], names=["laser_id", "X", "Y", "Z"])
    num_of_scenes = len(giant_df) / 72000
    dataframes = []

    for i in range(int(num_of_scenes)):
        start = i * 72000
        end = start + 72000
        dataframes.append(giant_df.iloc[start:end, :])

    return dataframes


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
    np.save("{}/{}_input.npy".format(save_here, object_names[object_choice]), input_data)
    np.save("{}/{}_output.npy".format(save_here, object_names[object_choice]), output)


def train_test_split(object_id, object_names, target, train_percent=0.7):
    """
        Function that creates the train and test split
    """
    # Input file and output file names
    file_name = "{}/{}_input.npy".format(target,object_names[object_id])
    output_file = "{}/{}_output.npy".format(target,object_names[object_id])

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

def optimize(num_iterations, train_batch_size, input_train, output_train, session, x, y_true, optimizer, accuracy, cost):

    # Start time
    start_time = datetime.now()

    # input and validation
    len_validation_from = len(input_train)-int(len(input_train)/7)
    
    input_train, output_train, val_input, val_output = train_validation(input_train, output_train, len_validation_from)
    
    # Accuracy and cost lists
    train_loss = []
    val_loss= []

    train_accu = []
    val_accu = []


    num_of_batches = math.ceil(len(input_train)/train_batch_size)

    # Converting the input into tensors
    x_batch1 = tf.convert_to_tensor(input_train)
    y_true_batch1 = tf.convert_to_tensor(output_train)

    # Creating the input_queue
    input_queue = tf.train.slice_input_producer([x_batch1, y_true_batch1])

    # Slicing the image
    sliced_x = input_queue[0]
    sliced_y = input_queue[1]

    # Batching the queue 
    x_batch2, y_true_batch2 = tf.train.batch([sliced_x, sliced_y], batch_size = train_batch_size, allow_smaller_final_batch=True)

    # Coordinating te multi threaded function
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = session)

    
    for i in range(num_of_batches*num_iterations):

        x_batch, y_true_batch = session.run([x_batch2, y_true_batch2])

        # Put the batch in the dict with the proper names
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of the training data
        session.run(optimizer, feed_dict=feed_dict_train)

        # printing status for every 10 iterations
        if i % num_of_batches == 0:

            count = 0
            val_acc = 0
            val_cost = 0
            
            for j in range(int(len(val_input)/100)):
                val_acc = val_acc+session.run(accuracy, feed_dict={x: np.array(val_input[j*100:(j+1)*100], dtype=np.float32), y_true: np.array(val_output[j*100:(j+1)*100], dtype=np.float32)})
                val_cost = val_cost+session.run(cost, feed_dict={x: np.array(val_input[j*100:(j+1)*100], dtype=np.float32), y_true: np.array(val_output[j*100:(j+1)*100], dtype=np.float32)})
                count+=1
            
            val_acc = val_acc/count
            val_cost = val_cost/count

            val_accu.appen(val_acc)
            val_loss.append(val_cost)

            # Calculating the train accuracy
            count = 0
            train_acc = 0
            train_cost = 0
            
            for j in range(int(len(input_train)/100)):
                train_acc = train_acc+session.run(accuracy, feed_dict={x: np.array(input_train[j*100:(j+1)*100], dtype=np.float32), y_true: np.array(output_train[j*100:(j+1)*100], dtype=np.float32)})
                train_cost = train_cost+session.run(cost, feed_dict={x: np.array(input_train[j*100:(j+1)*100], dtype=np.float32), y_true: np.array(output_train[j*100:(j+1)*100], dtype=np.float32)})
                count+=1
            
            train_acc = train_acc/count
            train_cost = train_cost/count

            train_accu.append(train_acc)
            train_loss.append(train_cost)
            
            print("---------")
            print(
                "Optimization Epochs: {0:>6}, Training Accuracy: {1:6.1%}, validation Accuracy: {2:6.1%}, training cost: {3}, val_cost: {4}".format(
                    (i/num_of_batches) + 1, train_acc, val_acc, train_cost, val_cost
                )
            )
        
    
    coord.request_stop()
    coord.join(threads) 
        
        

    # Ending time
    end_time = datetime.now()

    print("Time usage: {}".format(end_time - start_time))
    return train_accu, val_accu, train_loss, val_loss

def print_test_accuracy(test_input, test_output, session, y_true, y_pred_cls, x,  show_confusion_matrix=False):

    # number of images in the test -set
    num_test = len(test_input)

    # creating an empty array
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Starting index
    i = 0
    
    test_batch_size = 64

    while i < num_test:
        # J is the ending index
        j = min(i + test_batch_size, num_test)

        # get the images
        images = test_input[i:j]

        # Get the assiciated labels
        labels = test_output[i:j]

        # Feed the dict with the images and labels
        feed_dict = {x: images, y_true: labels}

        # Calculate the predicated class using TensorFlow
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = [np.argmax(i) for i in test_output]
    cls_true = np.array(cls_true)

    correct = cls_true == cls_pred
    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"

    print(msg.format(acc, correct_sum, num_test))

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        # Visualization().plot_confusion_matrix(cls_pred, cls_true)

def data_prep(save_here):
    # list of individual objects
    list_of_object_choice = list(range(29))
    list_of_object_choice.remove(22)

    # objects
    object_names = {0: 'Atm', 1: 'Bench', 2: 'BigSassafras', 3: 'BmwX5Simple', \
                    4: 'ClothRecyclingContainer', 5: 'Cypress', 6: 'DrinkingFountain',\
                    7: 'ElectricalCabinet', 8: 'EmergencyPhone', 9: 'FireHydrant',\
                    10: 'GlassRecyclingContainer', 11: 'IceFreezerContainer', 12: 'Mailbox',\
                    13: 'MetallicTrash', 14: 'MotorbikeSimple', 15: 'Oak', 16: 'OldBench',\
                    17: 'Pedestrian', 18: 'PhoneBooth', 19: 'PublicBin', 20: 'Sassafras',\
                    21: 'ScooterSimple', 22: 'set1', 23: 'ToyotaPriusSimple', 24: 'Tractor',\
                    25: 'TrashBin', 26: 'TrashContainer', 27: 'UndergroundContainer',\
                    28: 'WorkTrashContainer'}

    
    # Required variables
    num_linear_transformations = 4
    num_of_scenes = 50
    path_to_pkl = "../outliers.pkl"
    grid_size = 0.1
    num_clusters = 4
    img_length = 10
    img_height = 7
    
    # Folder names
    folder_path = "/home/samba693/DataChallenge/debs2019_initial_dataset"

    # Create the outliers
    get_outliers(folder_path)

    # Preparing the data
    for i in list_of_object_choice:
        prepare_and_save_input(i, object_names, folder_path, num_linear_transformations,\
                              num_of_scenes, path_to_pkl, grid_size,\
                              num_clusters, img_length, img_height, save_here)

if __name__ == "__main__":
    save_here = "../data/"
    data_prep(save_here)
    