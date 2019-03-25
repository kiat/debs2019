from allScences_gif import All_Scenes_Gif

def main():
    
    # Instantiating the object
    a = All_Scenes_Gif("/home/samba693/DataChallenge/debs2019_initial_dataset/BigSassafras/in.csv")

    # Removing the outliers
    object_points = a.remove_outliers()

    # plotting the two axis x = 0, y = 1, z = 2
    a.plot_two_axes(object_points,"BigSassafras",0,1)

    # Creating the gif
    a.create_gif("BigSassafras",0,1)

if __name__ == "__main__":
    main()