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
def get_object(path_to_data):
    # Read in the data set 
    outliers, list_of_dataframes = data.remove_outliers(path_to_data)
    clustered_dataframes = data.cluster(list_of_dataframes)
    object_points = data.object_points(clustered_dataframes)
    
    labeled_data = []

    # Adding the labels
    for i,j in zip(outliers, list_of_dataframes):
        i['labels'] = 0
        j['labels'] = 1
        i = i.append(j,ignore_index=True, sort=False)
        labeled_data.append(i)

    return labeled_data 

# Split the train and test cases
def split_train_test(objects):
    random.shuffle(objects)
    return objects[:30], objects[30:]

# Main function
def main():
    # Get the object points
    labeled_data = get_object(sys.argv[1])
    
    # visualizing the points
    data.visualize(labeled_data,2)

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