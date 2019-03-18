"""
@author: samba
"""

# Import the required libraries
import pandas as pd
import data
import sys
import random
import gan
import train
import evaluate

# Get the individual object points
def get_object(path_to_data, num_of_clusters):
    # Read in the data set 
    list_of_dataframes = data.remove_outliers(path_to_data)
    clustered_dataframes = data.cluster(list_of_dataframes,num_of_clusters)
    object_points = data.object_points(clustered_dataframes)
    return list_of_dataframes

# Split the train and test cases
def split_train_test(objects):
    random.shuffle(objects)
    return objects[:30], objects[30:]

# Main function
def main():
    # Get the object points
    object_points = get_object(sys.argv[1])
    
    # visualizing the points
    data.visualize(object_points,2)

    # Split train and test data
    # train_data, test_data = split_train_test(object_points)
    
    
    # train the data
    # model = train.fit(train_data)

    # save the model

    # test the data
    # results = evaluate.predict(test_data)


if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python train.py <path/to/file>")
        exit(-1)
    main()