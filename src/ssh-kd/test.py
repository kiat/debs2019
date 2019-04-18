def test():
    # Creating the session
    session, img_length, img_height, y_pred_cls, x = load_graph()

    # Folder names
    data_folder = "../dataset/test_data/in.csv"
    file_path1 = "{}/in.csv".format(folder_path1)
    
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
                        x)
        print(result)

if __name__ == "__main__":
    main()