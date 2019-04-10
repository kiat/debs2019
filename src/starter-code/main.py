import numpy as np

from .cnn2d import ClassifyWith2dCnn
from .cnn3d import ClassifyWith3dCnn
from .data import (
    read_file,
    input_nn,
    prepare_and_save_input,
    train_test_split,
    encode_output,
    flat_input,
)
from .visualization import Visualization
from .grouping import remove_outliers, helper_object_points, max_min
from .segmentation import *
from .evaluate import accuracy, precision, recall

object_names = {
    0: "Atm",
    1: "Bench",
    2: "BigSassafras",
    3: "BmwX5Simple",
    4: "ClothRecyclingContainer",
    5: "Cypress",
    6: "DrinkingFountain",
    7: "ElectricalCabinet",
    8: "EmergencyPhone",
    9: "FireHydrant",
    10: "GlassRecyclingContainer",
    11: "IceFreezerContainer",
    12: "Mailbox",
    13: "MetallicTrash",
    14: "MotorbikeSimple",
    15: "Oak",
    16: "OldBench",
    17: "Pedestrian",
    18: "PhoneBooth",
    19: "PublicBin",
    20: "Sassafras",
    21: "ScooterSimple",
    22: "set1",
    23: "ToyotaPriusSimple",
    24: "Tractor",
    25: "TrashBin",
    26: "TrashContainer",
    27: "UndergroundContainer",
    28: "WorkTrashContainer",
}

# Select the objects from above
user_choice = 23

# Path to dataset in file
folder_path = "/home/samba693/DataChallenge/debs2019_initial_dataset"
file_path = "{}/{}/in.csv".format(folder_path, object_names[user_choice])

# number of scenes to read in must be less than or equal to 50
num_of_scenes = 50

# Path to the outliers pkl file
outliers_path = "../object-net/outliers.pkl"

# 2-D image length and height
img_length = 10
img_height = 7

dataframes = read_file(file_path, num_of_scenes)
objects = remove_outliers(dataframes, num_of_scenes, outliers_path)

# Select only the desired scene and desired angle
scene_to_plot = objects[6]
view = 3

visual = Visualization()
# plotting the object with desired view
visual.visualize(scene_to_plot, view)

# Creating the animation
# The animation is created as a html file at src/visual/objectName_{num_of_scenes}_{view}
visual.visualize_50_html(
    num_of_scenes=50, view=view, objects=objects, object_names=object_names[view]
)

# change the view here
needed_view = 3

# Call the helper function to only get the object points
new_objects = []
num_clusters = 4

for i in objects:
    i = helper_object_points(i, num_clusters)
    new_objects.append(i)

# new_objects = objects

# calling the function for the range values
x_max, x_min, y_max, y_min = max_min(new_objects, img_length, img_height, needed_view)

# checking the range values
print("Image length = {}".format(img_length))
print(np.array(x_max) - np.array(x_min))
print("Image height = {}".format(img_height))
print(np.array(y_max) - np.array(y_min))

# select the image of choice between 0 to 49
img_choice = 34

# selecting the range values
x_range = [x_min[img_choice], x_max[img_choice]]
y_range = [y_min[img_choice], y_max[img_choice]]

# the scene to plot
scene_to_view = new_objects[img_choice]

# plotting the graph with the given range and if you want to change the view change the needed_view variable
visual.visualize(scene_to_view, needed_view, x_range, y_range)

# Variable for the gid size
grid_size = 0.1

# graph
visual.plot_grid(scene_to_view, grid_size, view, x_range, y_range)

# calling the function to get the input for the neural net to train
input_arr = input_nn(
    scene_to_view, x_range, y_range, grid_size, img_length, img_height, view
)
visual.simpleplt(input_arr)

# list of individual objects
list_of_object_choice = list(range(29))
list_of_object_choice.remove(22)

# list of objects
include_objects = list_of_object_choice

# load the data and append all the data into a single list both input and output data
input_train = []
input_test = []
output_train = []
output_test = []

for i in include_objects:
    # calling the function
    a, b, c, d = train_test_split(i, object_names)

    # adding them to the exsisting lists
    input_train += a
    input_test += b
    output_train += c
    output_test += d

# printing the number of training samples and their shapes
print("Size of:")
print("- Training-set:\t\t{}".format(len(input_train)))
print("- Test-set:\t\t{}".format(len(input_test)))

# Encoding and flattening the training data
train_out_encode, encoder, encoder1 = encode_output(output_train)
train_input_encode = flat_input(input_train)

# Encoding and Flattening the test data
test_out_encode, encoder2, encoder3 = encode_output(output_test)
test_input_encode = flat_input(input_test)

# Storing the images in one dimensional arraays of the length
img_size_flat = input_train[0].shape[0] * input_train[0].shape[1]

# Height and width of the image
img_shape = input_train[0].shape

# Number of classes
num_classes = len(include_objects)

# Number of color channels: 1 for grayscale
num_channels = 1

# Convolution Layer 1
filter_size1 = 5  # convolution filters are 5x5.
num_filters1 = 16  # There are 16 filters.

# Convolution Layer 2
filter_size2 = 5
num_filters2 = 16

# fully_connected Layer
fc_size = 128

cnn2d = ClassifyWith2dCnn()
cnn2d.sample_structure()

# optimize(1, train_batch_size, train_input_encode, train_out_encode)
# Testing after 10 iterations
cnn2d.print_test_accuracy(test_input_encode, test_out_encode, True)

# optimize(99, train_batch_size, train_input_encode, train_out_encode)
cnn2d.print_test_accuracy(test_input_encode, test_out_encode, True)

# optimize(4000, train_batch_size, train_input_encode, train_out_encode)
cnn2d.print_test_accuracy(test_input_encode, test_out_encode, True)

# TODO: Add up line 75 - 82 ***

# Folder names
folder_path1 = "/home/samba693/DataChallenge/debs2019_initial_dataset"
file_path1 = "{}/{}/in.csv".format(folder_path1, object_names[22])

# Read in the output to be predicted
output_path = "{}/{}/out.csv".format(folder_path1, object_names[22])
out_file = open(output_path, "r").readlines()
outfile = [i.rstrip() for i in out_file]
outfile = [i.split(",") for i in outfile]
outfile = [i[1:] for i in outfile]

outfile_list = []

for i in outfile:
    a = {}
    for j in range(0, len(i), 2):
        a[i[j]] = int(i[j + 1])
    outfile_list.append(a)

num_of_test_scenes = len(outfile_list)

# Read the input to be tested
test_dataframes = read_file(file_path1, num_of_test_scenes)

# Remove the outliers using the fixed values
test_outliers_dataframes = remove_outliers(
    test_dataframes, num_of_test_scenes, outliers_path
)

# TODO : Call segmentation here

output_dict_pred = convert_pred_to_dict(output_pred)

# Accuray
accuracy_ = []
precision_ = []
recall_ = []
for i, j in zip(output_dict_pred, outfile_list):
    accuracy_.append(accuracy(i, j))
    precision_.append(precision(i, j))
    recall_.append(recall(i, j))

acc = sum(accuracy_) / len(accuracy_)
prec = sum(precision_) / len(precision_)
reca = sum(recall_) / len(recall_)

# Printing the accuracy and recall
print("Accuracy = {}".format(acc))
print("Precision = {}".format(prec))
print("Recall = {}".format(reca))
