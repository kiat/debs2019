import tensorflow as tf
from cnn3d import ClassifyWith3dCnn
from pred import convert_pred_to_dict


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
def load_graph(path_to_model="../model/3dcnn.h5"):
    # variable
    img_length = 16
    img_height = 16

    # Creates the graph
    cnn_ = ClassifyWith3dCnn()
    cnn_.upload_model(path_to_model)

    return cnn_, img_length, img_height


# Remove the outliers, segment the data, predict the output and return json
def return_prediction(
    data_frame, session, object_names, img_length, img_height
):
    return convert_pred_to_dict(data_frame, session, object_names, img_length, img_height)
