import time
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(".."))       # path hack
from sklearn.cluster import DBSCAN, MeanShift, MiniBatchKMeans, KMeans
from utils.data_prep import read_file
from plugin.seg import remove_outliers

# DBSCAN
def dbscan_seg(data_frame):
    clustering = DBSCAN(eps=1,min_samples=16).fit(data_frame)
    return len(np.unique(clustering.labels_))-1

# MeanShift
def meanshift_seg(data_frame):
    clustering = MeanShift(bandwidth=2, bin_seeding=True, cluster_all=False, min_bin_freq=19, n_jobs=None, seeds=None).fit(data_frame)
    return len(np.unique(clustering.labels_))

# MiniBatchKMeans
def doClusteringWitkMiniBatchKmeans(data, min_cluster_number=10, max_cluster_number=50, Elbow_ratio = 1.02):

    Sum_of_squared_distances = []

    for k in range(min_cluster_number, max_cluster_number):
        if(data.shape[0] <= k):
            break
        km = MiniBatchKMeans(n_clusters=k, batch_size=int(np.size(data,0) * 1), max_iter=100, random_state=0)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)

    numberOfClusters = 1
    for i in range(1, len(Sum_of_squared_distances)):
        ratio=float((Sum_of_squared_distances[i-1])/Sum_of_squared_distances[i])
        # elbow ratio is an important parameter. 
        if(ratio < Elbow_ratio):
            numberOfClusters=i+1
            break
    # final run with large iterations 
    km = KMeans(n_clusters=numberOfClusters, max_iter=100, random_state=0)
    km = km.fit(data)
    
    return numberOfClusters

# KMeans
def kmeans_seg(data_frame):
    pass

# Count the number of actual clusters
def load_output(file_path):
    out_file = open(file_path, 'r').readlines()
    outfile = [i.rstrip() for i in out_file]
    outfile = [i.split(',') for i in outfile]
    outfile = [i[1:] for i in outfile]

    outfile_list = []
    out_cluster = []
    for i in outfile:
        a  = {}
        for j in range(0,len(i),2):
            a[i[j]] = int(i[j+1])
        out_cluster.append(sum(a.values()))
        
    return out_cluster

def cal_error(pred,original):
    error = np.array(pred)-np.array(original)
    rms = np.sum(error**2)/len(error)
    return rms

# main
def main():
    
    # read in the data frames
    # dataframes = read_file("../../dataset/test/debs2019_dataset2/in.csv")

    # remove outliers 
    # no_out = remove_outliers(dataframes,path_to_pkl="../data/outliers.pkl")        # returns the list of numpy arrays
    
    # read in the number of clusters
    out_clusters = load_output("../../dataset/test/debs2019_dataset2/in.csv")

    # record the time
    db_time = 0
    mean_time = 0
    mini_time = 0
    kmeans_time = 0

    # record the accuracy by comparing the number of clusters
    db_pred = []
    mean_pred = []
    mini_pred = []
    kmean_pred = []

    for j in range(500):
        print(j)
        j = pd.read_csv("../../dataset/test/debs2019_dataset2/in.csv",skipcols = j*72000, nrows = 72000,usecols=[1,2,3,4],columns=['laser_id','X','Y','Z'])
        i = remove_outliers(j,path_to_pkl="../data/outliers.pkl")[0]
        start = time.time()
        db_clusters = dbscan_seg(i)
        end = time.time()
        db_pred.append(db_clusters)
        db_time+=end-start

        start = time.time()
        mean_clusters = meanshift_seg()
        end = time.time()
        mean_pred.append(mean_clusters)
        mean_time+=end-start

        start = time.time()
        mini_clusters= minibatch_seg()
        end = time.time()
        mini_pred.append(mini_clusters)
        mini_time+=end-start

        # start = time.time()
        # kmeans_clusters = kmeans_seg()
        # end = time.time()
        # kmean_pred.append(kmeans_clusters)
        # kmeans_time+=end-start
    
    # Calculate the accuracy for each technique used
    db_acc = cal_accuracy(db_pred,out_clusters)
    mean_acc = cal_accuracy(mean_pred,out_clusters)
    mini_acc = cal_accuracy(mini_pred,out_clusters)
    # kmean_acc = cal_accuracy(kmean_pred,out_clusters)

    # write the calculated results to a file
    output = {'Method':['DBSCAN','MeanShift','MiniBatchKMeans'],
    'Time':[db_time,mean_time,mini_time],
    'Error':[db_acc,mean_acc,mini_acc]}

    df = pd.DataFrame(output)
    df.to_csv('seg_eval.csv')

if __name__ == "__main__":
    main()
