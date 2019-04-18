import tensorflow as tf

# object names
def object_names_func():
    # Object names
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

    return object_names
    
# create the model and restore the weights
def load_graph(path_to_model="../models/two_d_cnn.ckpt"):

    # variable
    img_length = 10
    img_height = 7

    # Creates the graph
    cnn2d = ClassifyWith2dCnn()
    cnn2d.sample_structure()

    # Create the saver object, start the session and restore the weights
    saver = tf.train.Saver()
    cnn2d.session = tf.Session()
    session = cnn2d.session
    session.run(tf.global_variables_initializer())
    saver.restore(session, path_to_model)
    y_pred_cls =  cnn2d.y_pred_cls_
    x = cnn2d.x
    
    return session, img_length, img_height, y_pred_cls, x

# Remove the outliers, segment the data, predict the output and return json
def return_prediction(data_frame, session, img_length, img_height, y_pred_cls, x):
    return convert_pred_to_dict(data_frame,session, img_length, img_height, y_pred_cls, x)