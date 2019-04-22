import tensorflow as tf

from cnn3d import ClassifyWith3dCnn
from data import train_test_split, encode_output, flat_input, optimize, print_test_accuracy
from segment import predict, convert_pred_to_dict
from data import read_file


# prepare the data fro train and test
def train_test(list_of_objects, object_names, target):
    input_train = []
    input_test = []
    output_train = []
    output_test = []

    for i in list_of_objects:
        # Calling the function
        a, b, c, d = train_test_split(i, object_names, target)

        # adding them to the exsisting lists
        input_train += a
        input_test += b
        output_train += c
        output_test += d

    return input_train, input_test, output_train, output_test


def return_prediction(data_frame, object_names, img_length, img_height):
    return predict(data_frame, object_names, img_length, img_height)


from functools import reduce


# Compute the accuracy
def accuracy(a, b):
    common_keys = set(a).intersection(b)
    all_keys = set(a).union(b)
    score = len(common_keys) / len(all_keys)  # key score
    if (score == 0):
        return score
    else:  # value score
        pred = {}
        for k in common_keys:
            pred[k] = b[k]
        # true_values_sum = reduce(lambda x,y:int(x)+int(y),a.values())
        all_keys = dict.fromkeys(all_keys, 0)
        for k in a.keys():
            all_keys.update({k: a[k]})
        for k in b.keys():
            all_keys.update({k: b[k]})
        true_values_sum = reduce(lambda x, y: int(x) + int(y), all_keys.values())
        pred_values_sum = reduce(lambda x, y: int(x) + int(y), pred.values())
        val_score = int(pred_values_sum) / int(true_values_sum)
        return (score + val_score) / 2  # avg


def train():
    # objects
    object_names = {0: 'Atm', 1: 'Bench', 2: 'BigSassafras', 3: 'BmwX5Simple',
                    4: 'ClothRecyclingContainer', 5: 'Cypress', 6: 'DrinkingFountain',
                    7: 'ElectricalCabinet', 8: 'EmergencyPhone', 9: 'FireHydrant',
                    10: 'GlassRecyclingContainer', 11: 'IceFreezerContainer', 12: 'Mailbox',
                    13: 'MetallicTrash', 14: 'MotorbikeSimple', 15: 'Oak', 16: 'OldBench',
                    17: 'Pedestrian', 18: 'PhoneBooth', 19: 'PublicBin', 20: 'Sassafras',
                    21: 'ScooterSimple', 22: 'set1', 23: 'ToyotaPriusSimple', 24: 'Tractor',
                    25: 'TrashBin', 26: 'TrashContainer', 27: 'UndergroundContainer',
                    28: 'WorkTrashContainer'}

    # target
    target = "../data"
    num_classes = 28

    # list of individual objects
    list_of_object_choice = list(range(num_classes))
    # list_of_object_choice.remove(22)

    # Prepare the data
    input_train, input_test, output_train, output_test = train_test(list_of_object_choice, object_names, target)

    # Encoding and flattening the training data
    train_out_encode, encoder, encoder1 = encode_output(output_train)
    train_input_encode = flat_input(input_train)

    # Encoding and Flattening the test data
    test_out_encode, encoder2, encoder3 = encode_output(output_test)
    test_input_encode = flat_input(input_test)

    print(train_input_encode.shape)
    print(train_out_encode.shape)
    cnn_ = ClassifyWith3dCnn(train_input_encode, train_out_encode, test_input_encode, test_out_encode, num_classes)
    cnn_.cnn_initiate()

    folder_path1 = "../../../dataset"
    file_path1 = "{}/Set2/in.csv".format(folder_path1)

    out_path = '{}/Set2/out.csv'.format(folder_path1)
    out_file = open(out_path, 'r').readlines()

    # Read in the test data
    data_frames = read_file(file_path1)
    accuracy_ = []
    for it_, reconstructed_scene in enumerate(data_frames):
        result = return_prediction(reconstructed_scene, object_names, img_length=16, img_height=16)
        result_dict = cnn_.evaluate(result)
        result_dict = convert_pred_to_dict(result_dict, object_names)
        out_ = {}
        line_sum = out_file[it_].strip().split(',')
        for val in range(1, len(line_sum), 2):
            out_[line_sum[val]] = line_sum[val + 1]
        accuracy_.append(accuracy(result_dict, out_))
        print(accuracy_[-1])
    print('overall acc:', sum(accuracy_) / len(accuracy_))


if __name__ == "__main__":
    from datetime import datetime

    starting = datetime.now()
    print("Started at:", starting)
    train()
    print("Ended at:", datetime.now() - starting)
