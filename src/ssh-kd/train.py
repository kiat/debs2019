#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt

from plugin.model import ClassifyWith2dCnn
from utils.data_prep import train_test_split
from plugin.encode import flat_input, encode_output
from plugin.load_model import object_names_func
from utils.optimize import optimize, print_test_accuracy

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

def load_data(target="data"):

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

    # create the tensorflow graph
    cnn2d = ClassifyWith2dCnn()
    cnn2d.sample_structure()
    cnn2d.session = tf.Session()
    session = cnn2d.session
    x = cnn2d.x
    y_true = cnn2d.y_true
    y_pred_cls = cnn2d.y_pred_cls_
    optimizer = cnn2d.optimizer
    accuracy = cnn2d.accuracy
    cost = cnn2d.cost
    prob = cnn2d.prob

    # train the data
    session.run(tf.global_variables_initializer())
    train_batch_size = 32

    train_acc, val_acc, train_cost, val_cost = optimize(
        10,
        train_batch_size,
        train_input_encode,
        train_out_encode,
        session,
        x,
        y_true,
        optimizer,
        accuracy,
        cost,
        prob
    )

    # fig = plt.figure()
    # plt.plot(train_acc)
    # plt.plot(val_acc)
    # fig.savefig("model/accuracy.png")
    # plt.clf()

    # fig = plt.figure()
    # plt.plot(train_cost)
    # plt.plot(val_cost)
    # fig.savefig("model/loss.png")
    # plt.clf()

    # saving the model
    path_to_model = "model/two_d_cnn_proj.ckpt"
    save_model(session, path_to_model)

    # plot the test accuracy in between
    print_test_accuracy(
        test_input_encode, test_out_encode, session, y_true, y_pred_cls, x
    )

    # Closing the tensorflow session
    session.close()


if __name__ == "__main__":
    train()
