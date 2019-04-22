#     folder_path1 = "../../../dataset"
#     file_path1 = "{}/Set2/in.csv".format(folder_path1)

#     out_path = '{}/Set2/out.csv'.format(folder_path1)
#     out_file = open(out_path, 'r').readlines()

#     # Read in the test data
#     data_frames = read_file(file_path1)
#     accuracy_ = []
#     for it_, reconstructed_scene in enumerate(data_frames):
#         result = return_prediction(reconstructed_scene, object_names, img_length=16, img_height=16)
#         result_dict = cnn_.evaluate(result)
#         result_dict = convert_pred_to_dict(result_dict, object_names)
#         out_ = {}
#         line_sum = out_file[it_].strip().split(',')
#         for val in range(1, len(line_sum), 2):
#             out_[line_sum[val]] = line_sum[val + 1]
#         accuracy_.append(accuracy(result_dict, out_))
#         print(accuracy_[-1])
#     print('overall acc:', sum(accuracy_) / len(accuracy_))

# def return_prediction(data_frame, object_names, img_length, img_height):
#     return predict(data_frame, object_names, img_length, img_height)


# # Compute the accuracy
# def accuracy(a, b):
#     common_keys = set(a).intersection(b)
#     all_keys = set(a).union(b)
#     score = len(common_keys) / len(all_keys)  # key score
#     if (score == 0):
#         return score
#     else:  # value score
#         pred = {}
#         for k in common_keys:
#             pred[k] = b[k]
#         # true_values_sum = reduce(lambda x,y:int(x)+int(y),a.values())
#         all_keys = dict.fromkeys(all_keys, 0)
#         for k in a.keys():
#             all_keys.update({k: a[k]})
#         for k in b.keys():
#             all_keys.update({k: b[k]})
#         true_values_sum = reduce(lambda x, y: int(x) + int(y), all_keys.values())
#         pred_values_sum = reduce(lambda x, y: int(x) + int(y), pred.values())
#         val_score = int(pred_values_sum) / int(true_values_sum)
#         return (score + val_score) / 2  # avg


import os
from datetime import datetime

from load_model import load_graph, return_prediction, object_names_func
from custom_data import read_file
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
    session, img_length, img_height = load_graph()
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
                    img_height)
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
