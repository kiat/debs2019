import numpy as np
import math
import matplotlib.pyplot as plt

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