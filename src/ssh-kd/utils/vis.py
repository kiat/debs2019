import numpy as np
import math
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.abspath(".."))

from plugin.load_model import object_names_func

def plot_conv_weights(w, input_channel=0, file_name="../model/weights.png" ):

    weight_min = np.min(w)
    weight_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        
        # Only plot the valid filter-weights.
        if i<num_filters:
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=weight_min, vmax=weight_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    plt.savefig(file_name)


def plot_conv_layer(values , file_name="../model/conv.png" ):
    
    # Number of filters used in the conv layer
    num_filters = values.shape[3]

    # Number of grids to plot.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    plt.savefig(file_name)

def plot_input_images():

    object_names = object_names_func()
    list_of_object_choice = list(range(29))
    list_of_object_choice.remove(22)

    # reading in the data of the desired object
    for j in list_of_object_choice:
        desired_object = j
        file_name = "../data/{}_input.npy".format(object_names[desired_object])
        to_plot = np.load(file_name)


        # plotting the object
        fig = plt.figure(figsize=(100,100))
        colums = 20
        rows = 20
        count = 1

        for i in to_plot:    
            fig.add_subplot(rows, colums, count)
            count+=1
            plt.imshow(i)

        plt.savefig("../model/{}.png".format(object_names[desired_object]))
        plt.clf()

if __name__ == "__main__":
    plot_input_images()