# Debs 2019

# Packages
import pandas
import numpy as np
import time

# Visualization
from voxelgrid import VoxelGrid

# Data Grouping
from sklearn.neighbors import KNeighborsClassifier
from ConvNN import CNNClassify

# Training:
def training():
    train_x = []
    y_train = []
    for filen in ['Atm', 'Pedestrian']:
        print("Grouping {}".format(filen))
        directory = 'dataset/{}'.format(filen)
        in_path = '{}/in.csv'.format(directory)

        out_path = '{}/out.csv'.format(directory)
        out_file = open(out_path, 'r').readlines()

        for line_i in range(len(out_file) - 20):
            line_sum = out_file[line_i].strip().split(',')[1]
            y_train.append(line_sum)

        x, y, z = [], [], []
        # Reading in the data for one scene at a time
        for j in range(len(out_file) - 20):
            df = pandas.read_csv(in_path, skiprows=j * 72000, nrows=72000, names=["timestamp", "id", "x", "y", "z"])
            df['r_sq'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
            df['r'] = np.sqrt(df['r_sq']).round(1)
            df['freq'] = df.groupby(['id', 'r'])['r'].transform('count')
            for i in range(64):
                maxfreq = df[(df['id'] == i) & (df['r'] != 0)]['freq'].max()
                while maxfreq > 100:
                    df.drop(df[(df['id'] == i) & (df['freq'] == maxfreq)].index, inplace=True)
                    maxfreq = df[(df['id'] == i) & (df['r'] != 0)]['freq'].max()
                df.drop(df[(df['id'] == i) & (df['r'] == 0)].index, inplace=True)
            df = df[['x', 'y', 'z']]
            for i in range(df.shape[0]+1, 760):
                df.loc[i]=(df.mean()['x'],df.mean()['y'],df.mean()['z'])
            voxel_grid = VoxelGrid(df.values, x_y_z=[16, 16, 16])
            train_x.append(voxel_grid.vector.reshape(-1) / np.max(voxel_grid.vector))
            # print('done {}'.format(j))
    return np.array(train_x), y_train


def testing():
    # Testing:
    test_x = []
    y_test = []
    for filen in ['Atm', 'Pedestrian']:
        print("Grouping {}".format(filen))
        directory = 'dataset/{}'.format(filen)
        in_path = '{}/in.csv'.format(directory)

        out_path = '{}/out.csv'.format(directory)
        out_file = open(out_path, 'r').readlines()

        for line_i in range(len(out_file) - 20, len(out_file)):
            line_sum = out_file[line_i].strip().split(',')[1]
            y_test.append(line_sum)

        x, y, z = [], [], []
        # Reading in the data for one scene at a time
        for j in range(len(out_file) - 20, len(out_file)):
            df = pandas.read_csv(in_path, skiprows=j * 72000, nrows=72000, names=["timestamp", "id", "x", "y", "z"])
            df['r_sq'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
            df['r'] = np.sqrt(df['r_sq']).round(1)
            df['freq'] = df.groupby(['id', 'r'])['r'].transform('count')
            for i in range(64):
                maxfreq = df[(df['id'] == i) & (df['r'] != 0)]['freq'].max()
                while maxfreq > 100:
                    df.drop(df[(df['id'] == i) & (df['freq'] == maxfreq)].index, inplace=True)
                    maxfreq = df[(df['id'] == i) & (df['r'] != 0)]['freq'].max()
                df.drop(df[(df['id'] == i) & (df['r'] == 0)].index, inplace=True)
            df = df[['x', 'y', 'z']]
            # standardizing all the point cloud to a similar shape,using mean to fill the input
            for i in range(df.shape[0]+1,760):
                df.loc[i] = (df.mean()['x'],df.mean()['y'],df.mean()['z'])
            # using voxelation to convert 3d point cloud to 2d structure.
            voxel_grid = VoxelGrid(df.values, x_y_z=[16, 16, 16])
            test_x.append(voxel_grid.vector.reshape(-1) / np.max(voxel_grid.vector))
            # print('done {}'.format(j),df.shape)
    return np.array(test_x), y_test


def KNNclassif(Xtrain, Ytrain, Xtest, Ytest):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(Xtrain, Ytrain)
    y_pred = neigh.predict(Xtest)
    print("Predicted value of y: ", y_pred)
    print("Accuracy score of KNN: {}".format(neigh.score(Xtest, Ytest)*100))
    print("----------------------")


def CNNclassif(Xtrain, Ytrain, Xtest, Ytest):
    ytrain = [0 if y=="Atm" else 1 for y in Ytrain]
    ytest = [0 if y=="Atm" else 1 for y in Ytest]
    cnn = CNNClassify(Xtrain, ytrain, Xtest, ytest)
    cnn.cnn_initiate()
    print("----------------------")

if __name__ == '__main__':
    start_time = time.time()
    Xtrain, Ytrain = training()
    Xtest, Ytest = testing()
    KNNclassif(Xtrain, Ytrain, Xtest, Ytest)
    CNNclassif(Xtrain, Ytrain, Xtest, Ytest)
    print("Time Taken for Execution {}m".format((time.time() - start_time)/60))
