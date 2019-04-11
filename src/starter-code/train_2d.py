import tensorflow as tf

from cnn2d import ClassifyWith2dCnn
from data import train_test_split, encode_output, flat_input, optimize

# saving the model
def save_model(session, path_to_model):
    
    # Saving the model
    saver = tf.train.Saver()
    saver.save(session,path_to_model)

# prepare the data fro train and test
def train_test(list_of_objects, object_names, target):
    input_train = []
    input_test = []
    output_train = []
    output_test = []

    for i in list_of_objects:
        # Calling the function
        a,b,c,d = train_test_split(i, object_names, target)

        # adding them to the exsisting lists
        input_train += a
        input_test += b
        output_train += c
        output_test += d

    return input_train, input_test, output_train, output_test

def train():
    # objects
    object_names = {0: 'Atm', 1: 'Bench', 2: 'BigSassafras', 3: 'BmwX5Simple', \
                    4: 'ClothRecyclingContainer', 5: 'Cypress', 6: 'DrinkingFountain',\
                    7: 'ElectricalCabinet', 8: 'EmergencyPhone', 9: 'FireHydrant',\
                    10: 'GlassRecyclingContainer', 11: 'IceFreezerContainer', 12: 'Mailbox',\
                    13: 'MetallicTrash', 14: 'MotorbikeSimple', 15: 'Oak', 16: 'OldBench',\
                    17: 'Pedestrian', 18: 'PhoneBooth', 19: 'PublicBin', 20: 'Sassafras',\
                    21: 'ScooterSimple', 22: 'set1', 23: 'ToyotaPriusSimple', 24: 'Tractor',\
                    25: 'TrashBin', 26: 'TrashContainer', 27: 'UndergroundContainer',\
                    28: 'WorkTrashContainer'}
    
    # target
    target = "../data"

    # list of individual objects
    list_of_object_choice = list(range(29))
    list_of_object_choice.remove(22)

    # Prepare the data
    input_train, input_test, output_train, output_test = train_test(list_of_object_choice, object_names, target)

    # Encoding and flattening the training data
    train_out_encode, encoder, encoder1 = encode_output(output_train)
    train_input_encode = flat_input(input_train)

    # Encoding and Flattening the test data
    test_out_encode, encoder2, encoder3 = encode_output(output_test)
    test_input_encode = flat_input(input_test)

    # create the tensorflow graph
    cnn2d = ClassifyWith2dCnn()
    cnn2d.sample_structure()
    cnn2d.session = tf.Session()
    session = cnn2d.session
    x = cnn2d.x
    y_true = cnn2d.y_true
    optimizer = cnn2d.optimizer
    accuracy = cnn2d.accuracy

    # train the data
    session.run(tf.global_variables_initializer())
    train_batch_size = 32

    optimize(2000, train_batch_size, train_input_encode, train_out_encode, session,x, y_true, optimizer, accuracy)

    # saving the model
    path_to_model = "../dummy/two_d_cnn.ckpt"
    save_model(session, path_to_model)

    # plot the test accuracy in between

if __name__ == "__main__":
    train()
    