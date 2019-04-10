# Packages
import pandas as pd
import numpy as np

# Libraries for neural net training and evaluation
import tensorflow as tf
from .visualization import Visualization


class ClassifyWith2dCnn(object):
    def __init__(self):
        self.tf.__version__ = tf.__version__
        self.session = tf.Session()
        self.session_run = self.session.run(tf.global_variables_initializer())
        self.train_batch_size = 32
        # Counter for the total number of iterations performed so far
        self.total_iterations = 0
        self.test_batch_size = 64

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(
        self,
        input_layer,
        num_input_channels,
        filter_size,
        num_filters,
        use_pooling=True,
    ):
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        # Create new weights with the shape
        weights = self.new_weights(shape=shape)
        # create the new biases of each filter
        biases = self.new_biases(length=num_filters)
        # creating the layer
        layer = tf.nn.conv2d(
            input=input_layer, filter=weights, strides=[1, 1, 1, 1], padding="SAME"
        )
        # adding the bias to the layer
        layer += biases
        # Create the max pooling layer
        if use_pooling:
            layer = tf.nn.max_pool(
                value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
            )
        # Adding relu
        layer = tf.nn.relu(layer)
        # return the layer and weights. Weights are used to plot the weights later
        return layer, weights

    def flatten_layer(self, layer):
        # get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assusmed to be
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height* img_width * num_of channels
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features]
        layer_flat = tf.reshape(layer, [-1, num_features])

        # returning the flattened layer
        return layer_flat, num_features

    def new_fc_layer(self, input_layer, num_inputs, num_outputs, use_relu=True):
        # create new weights and biases
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # calculating the layer
        layer = tf.matmul(input_layer, weights) + biases

        # check if is relu
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

    def sample_structure(self):
        x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="x")
        x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")
        y_true_cls = tf.argmax(y_true, axis=1)
        layer_conv1, weights_conv1 = self.new_conv_layer(
            input_layer=x_image,
            num_input_channels=num_channels,
            filter_size=filter_size1,
            num_filters=num_filters1,
            use_pooling=True,
        )

        layer_conv2, weights_conv2 = self.new_conv_layer(
            input_layer=layer_conv1,
            num_input_channels=num_filters1,
            filter_size=filter_size2,
            num_filters=num_filters2,
            use_pooling=True,
        )
        layer_flat, num_features = self.flatten_layer(layer_conv2)
        layer_fc1 = self.new_fc_layer(
            input_layer=layer_flat,
            num_inputs=num_features,
            num_outputs=fc_size,
            use_relu=True,
        )
        layer_fc2 = self.new_fc_layer(
            input_layer=layer_fc1,
            num_inputs=fc_size,
            num_outputs=num_classes,
            use_relu=False,
        )
        y_pred = tf.nn.softmax(layer_fc2)
        y_pred_cls = tf.argmax(y_pred, axis=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=layer_fc2, labels=y_true
        )
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def random_batch(self, input_train, output_train, batch_size):
        # zipping the input and output
        zipped_data = list(zip(input_train, output_train))

        # Shuffle them
        random.shuffle(zipped_data)

        # select the elements equal to batch_size
        zipped_data = zipped_data[:batch_size]

        # unzip the elements
        to_return = list(zip(*zipped_data))
        x_batch = np.array(to_return[0], dtype=np.float32)
        y_batch = np.array(to_return[1], dtype=np.float32)

        # return the elements
        return x_batch, y_batch

    def optimize(self, num_iterations, input_train, output_train):

        global total_iterations

        # Start time
        start_time = datetime.now()

        for i in range(total_iterations, total_iterations + num_iterations):

            # get the batch of the training examples
            x_batch, y_true_batch = self.random_batch(
                input_train, output_train, batch_size=self.train_batch_size
            )

            # Put the batch in the dict with the proper names
            feed_dict_train = {x: x_batch, y_true: y_true_batch}

            # Run the optimizer using this batch of the training data
            self.session.run(optimizer, feed_dict=feed_dict_train)

            # printing status for every 10 iterations
            if i % 100 == 0:
                # Calculate the accuracy on the training set
                acc = self.session.run(accuracy, feed_dict=feed_dict_train)
                print(
                    "Optimization Iteration: {0:>6}, Training Accuracy: {1:6.1%}".format(
                        i + 1, acc
                    )
                )

        # updating the total number of iterations performed
        total_iterations += num_iterations

        # Ending time
        end_time = datetime.now()

        print("Time usage: {}".format(end_time - start_time))

    def print_test_accuracy(self, test_input, test_output, show_confusion_matrix=False):

        # number of images in the test -set
        num_test = len(test_input)

        # creating an empty array
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Starting index
        i = 0

        while i < num_test:
            # J is the ending index
            j = min(i + self.test_batch_size, num_test)

            # get the images
            images = test_input[i:j]

            # Get the assiciated labels
            labels = test_output[i:j]

            # Feed the dict with the images and labels
            feed_dict = {x: images, y_true: labels}

            # Calculate the predicated class using TensorFlow
            cls_pred[i:j] = self.session.run(y_pred_cls, feed_dict=feed_dict)

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
            Visualization().plot_confusion_matrix(cls_pred, cls_true)
