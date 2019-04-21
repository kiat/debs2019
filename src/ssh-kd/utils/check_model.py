import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(".."))
from utils.vis import plot_conv_weights, plot_conv_layer
from plugin.load_model import load_graph, object_names_func
from train import load_data

def check_model():
    
    # Creating the session
    session, img_length, img_height, y_pred_cls, x, weights1, weights2, conv1, conv2 = load_graph(True,"../model/two_d_cnn_proj.ckpt")
    
    # object names
    object_names = object_names_func()
    list_of_objects = list(range(29))
    list_of_objects.remove(22)

    # plotting the weights
    plot_conv_weights(session.run(weights1), 0,'../model/weights1.png')
    plot_conv_weights(session.run(weights2), 0,'../model/weights2.png')




    # Select some images after reading in the data
    train_input_encode, train_out_encode, test_input_encode, test_out_encode = load_data("../data")

    image = train_input_encode[1000]
    test_object = object_names[list_of_objects[np.argmax(train_out_encode[1000])]]

    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer1
    values = session.run(conv1, feed_dict=feed_dict)
    plot_conv_layer(values, '../model/{}_1.png'.format(test_object))

    # Calculate and retrieve the output values of the layer2
    values = session.run(conv2, feed_dict=feed_dict)
    plot_conv_layer(values, '../model/{}_2.png'.format(test_object))

    print("Object = {}".format(test_object))
    

if __name__ == "__main__":
    check_model()