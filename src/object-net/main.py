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
def get_object(dataframes):
    clustered_dataframes = data.cluster(dataframes)
    object_points = data.object_points(clustered_dataframes)
    return object_points

# Split the train and test cases
def split_train_test(objects):
    random.shuffle(objects)
    return objects[:30], objects[30:]

# Main function
def main():
    # Read in the data set 
    list_of_dataframes = data.remove_outliers(sys.argv[1])
    
    # Get the object points
    object_points = get_object(list_of_dataframes)

    # Split train and test data
    train_data, test_data = split_train_test(object_points)

    # train the data
    # model = train.fit(train_data)

    # test the data
    # results = evaluate.predict(test_data)


if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python train.py <path/to/file>",file=sys.stderr)
        exit(-1)
    main()