import os
from datetime import datetime

from plugin.load_model import load_graph, return_prediction, object_names_func
from utils.data_prep import read_file
from utils.eval import accuracy, precision, recall

def format_output(output_path):

    # Read in the output to be predicted
    out_file = open(output_path, 'r').readlines()
    outfile = [i.rstrip() for i in out_file]
    outfile = [i.split(',') for i in outfile]
    outfile = [i[1:] for i in outfile]

    outfile_list = []

    for i in outfile:
        a  = {}
        for j in range(0,len(i),2):
            a[i[j]] = int(i[j+1])
        outfile_list.append(a)

    return outfile_list

def test():
    # Creating the session
    session, img_length, img_height, y_pred_cls, x, weights1, weights2, conv1, conv2 = load_graph()
    object_names = object_names_func()

    # Folder names
    dataset_dir = "../dataset/test/"
    actual_output = []
    pred_output = []
    
    # Checking the total time for testing
    start_time = datetime.now()

    for _set in os.listdir(dataset_dir):
        
        if os.path.isdir(os.path.join(dataset_dir,_set)):

            infile_path = os.path.join(dataset_dir, "{}/in.csv".format(_set))
            outfile_path = os.path.join(dataset_dir, "{}/out.csv".format(_set))

            # Read in the test data
            data_frames = read_file(infile_path)
            actual_output = actual_output + format_output(outfile_path)

            for reconstructed_scene in data_frames:
                result = return_prediction(
                    reconstructed_scene,
                    session,
                    object_names,
                    img_length,
                    img_height,
                    y_pred_cls,
                    x
                )
                pred_output.append(result)
    
    return actual_output, pred_output, datetime.now()-start_time

def metrics(actual_output, pred_output):
    
    # variables to return
    acc = 0
    prec = 0
    reca = 0

    for i,j in zip(actual_output, pred_output):
        acc = acc + accuracy(i,j)
        prec = prec + precision(i,j)
        reca = reca + recall(i,j)
    
    return acc, prec, reca


if __name__ == "__main__":
    actual, prediction, time_taken = test()
    
    total_scenes = len(prediction) 

    print("The total time taken for {} scenes  = {}".format(total_scenes, time_taken))
    
    acc, prec, reca = metrics(actual, prediction)
    
    print(" Accuracy = {} \n Precision = {} \n Recall = {}".format(acc/total_scenes, prec/total_scenes, reca/total_scenes))
