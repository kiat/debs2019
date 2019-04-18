import numpy as np
import tensorflow as tf

"""
    This class creates an  tensor flow graph 2_D_CNN graph
"""
class ClassifyWith2dCnn(object):
    def __init__(
        self, 
        img_shape=(70,100), 
        num_classes = 28,
        num_channels = 1, 
        filter_size = 5, 
        number_of_filters = 16, 
        fc_size = 128
        ):

        # To see the version of the tensorflow
        self.__version__ = tf.__version__

        # Variable to store the tf.Session and input and output layer of the graph
        self.session = None
        self.y_pred_cls_ = None
        self.x = None
        self.y_true = None
        self.accuracy = None
        self.optimizer = None
        self.cost = None
        
        # Varaibles to give the dimensions for the layers of graph
        self.img_shape = img_shape
        self.img_size_flat = img_shape[0]*img_shape[1]
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.filter_size1 = filter_size
        self.filter_size2 = filter_size
        self.num_filters1 = number_of_filters
        self.num_filters2 = number_of_filters
        self.fc_size = fc_size

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self,length):
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

    # Drop out layer
    def new_drop_out(self, input_layer, num_inputs, num_outputs, use_relu=True):
        # Create the new weights and biases
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Create the drop out layer
        dropped = tf.nn.dropout(input_layer, keep_prob = 0.5)

        # calculating the layer
        layer = tf.matmul(dropped, weights) + biases

        # check if is relu
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

    def sample_structure(self):
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name="x")
        x_image = tf.reshape(x, [-1, self.img_shape[0], self.img_shape[1], self.num_channels])
        y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="y_true")
        y_true_cls = tf.argmax(y_true, axis=1)
        layer_conv1, weights_conv1 = self.new_conv_layer(
            input_layer=x_image,
            num_input_channels=self.num_channels,
            filter_size=self.filter_size1,
            num_filters=self.num_filters1,
            use_pooling=True,
        )

        layer_conv2, weights_conv2 = self.new_conv_layer(
            input_layer=layer_conv1,
            num_input_channels=self.num_filters1,
            filter_size=self.filter_size2,
            num_filters=self.num_filters2,
            use_pooling=True,
        )

        # layer_conv3, weights_conv3 = self.new_conv_layer(
        #     input_layer=layer_conv2,
        #     num_input_channels=self.num_filters1,
        #     filter_size=self.filter_size2,
        #     num_filters=self.num_filters2,
        #     use_pooling=False,
        # )

        # layer_conv4, weights_conv4 = self.new_conv_layer(
        #     input_layer=layer_conv3,
        #     num_input_channels=self.num_filters1,
        #     filter_size=self.filter_size2,
        #     num_filters=self.num_filters2,
        #     use_pooling=True,
        # )

        layer_flat, num_features = self.flatten_layer(layer_conv2)
        layer_fc1 = self.new_fc_layer(
            input_layer=layer_flat,
            num_inputs=num_features,
            num_outputs=self.fc_size,
            use_relu=True,
        )
        layer_dropout = tf.nn.dropout(layer_fc1 ,keep_prob=0.8)
        layer_fc2 = self.new_fc_layer(
            input_layer=layer_dropout,
            num_inputs=self.fc_size,
            num_outputs=self.num_classes,
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

        self.x = x
        self.y_pred_cls_ = y_pred_cls
        self.y_true = y_true
        self.accuracy = accuracy
        self.optimizer = optimizer
        self.cost = cost
        