import os
from plugin.load_model import load_graph, return_prediction, object_names_func
from utils.data_prep import read_file


def test():
    # Creating the session
    session, img_length, img_height, y_pred_cls, x = load_graph()
    object_names = object_names_func()

    # Folder names
    dataset_dir = "../dataset/test/"
    for _set in os.listdir(dataset_dir):
        print(_set)
        file_path1 = os.path.join(dataset_dir, "{}/in.csv".format(_set))

        # Read in the test data
        data_frames = read_file(file_path1)

        for reconstructed_scene in data_frames:
            result = return_prediction(
                reconstructed_scene,
                session,
                object_names,
                img_length,
                img_height,
                y_pred_cls,
                x,
            )
            print(result)


if __name__ == "__main__":
    test()
