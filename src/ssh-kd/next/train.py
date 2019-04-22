import tensorflow as tf
import matplotlib.pyplot as plt

from cnn3d import ClassifyWith3dCnn
from custom_data import train_test_split
from encode import flat_input, encode_output
from load_model import object_names_func


# saving the model
def save_model(session, path_to_model):

    # Saving the model
    saver = tf.train.Saver()
    saver.save(session, path_to_model)


# prepare the data fro train and test
def train_test(list_of_objects, object_names, target):
    input_train = []
    input_test = []
    output_train = []
    output_test = []

    for i in list_of_objects:
        # Calling the function
        a, b, c, d = train_test_split(i, object_names, target)

        # adding them to the exsisting lists
        input_train += a
        input_test += b
        output_train += c
        output_test += d

    return input_train, input_test, output_train, output_test


def load_data(target="../data"):

    # list of individual objects
    list_of_object_choice = list(range(29))
    list_of_object_choice.remove(22)
    object_names = object_names_func()
    
    # Prepare the data
    input_train, input_test, output_train, output_test = train_test(
        list_of_object_choice, object_names, target
    )

    # Encoding and flattening the training data
    train_out_encode, encoder, encoder1 = encode_output(output_train)
    train_input_encode = flat_input(input_train)

    # Encoding and Flattening the test data
    test_out_encode, encoder2, encoder3 = encode_output(output_test)
    test_input_encode = flat_input(input_test)

    return train_input_encode, train_out_encode, test_input_encode, test_out_encode


def train():

    train_input_encode, train_out_encode, test_input_encode, test_out_encode = load_data()
    num_classes = 28
    # create the tensorflow graph
    cnn_ = ClassifyWith3dCnn(num_classes)
    cnn_.set_inputs(train_input_encode, train_out_encode, test_input_encode, test_out_encode, num_classes)
    cnn_.cnn_initiate()


if __name__ == "__main__":
    from datetime import datetime

    starting = datetime.now()
    print("Started at:", starting)
    train()
    print("Ended at:", datetime.now() - starting)
