import sys

from allScences_gif import All_Scenes_Gif

def main():
    if len(sys.argv) <= 4:
        print("python3.7 gif_creator.py /home/kia/Collected-Datasets/DEBS2019/debs2019_initial_dataset/ClothRecyclingContainer/in.csv ClothRecyclingContainer  0 1  2 ")
        print("plotting the 3d x = 0, y = 1, z = 2")
        return
    


    # Instantiating the object
    a = All_Scenes_Gif(sys.argv[1])

    # Removing the outliers
    object_points = a.remove_outliers()

    # plotting the two axis x = 0, y = 1, z = 2
    a.plot_two_axes(object_points, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

    # Creating the gif
    a.create_gif(sys.argv[2],0,1,2)

if __name__ == "__main__":
    main()
