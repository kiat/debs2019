"""
Note: Implemented with old implementation of remove_outliers.
"""
# packages
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("../ssh-kd"))
from plugin.seg import remove_outliers, helper_object_points
from utils.data_prep import read_file, get_outliers
import math
import matplotlib.pyplot as plt


def get_y(x, angle):
    y =[u * math.tan(math.radians(angle)) for u in x]
    return y


def get_x(x, angle):
    y =[float(u / math.tan(math.radians(angle))) for u in x]
    return y


def get_exceptions(p,val,point):
    response = []
    old_p = 0
    for an in [p, val]:
        if point[an] > 0 and point[an]< 90:
            x = [0.0, 15.0, 30.0]
            y = get_y(x, point[an])
            if point[p]<point[val] and an == p:
                z = get_y(x, 90)
                response.append((x,y,z))
            elif an == p:
                z = get_y(x, 0)
                response.append((x,z,y))
            else:
                response.append((x,old_p,y))
            if an == p:
                old_p = z
        elif point[an] > 90 and point[an]< 180:
            x = [0.0, -15.0, -30.0]
            y = get_y(x, point[an])
            if point[p]<point[val] and an == p:
                z = get_y(x, 180)
                response.append((x,y,z))
            elif an==p:
                z = get_y(x, 90)
                response.append((x,z,y))
            else:
                response.append((x,old_p,y))
            if an == p:
                old_p = z
        elif point[an] > 180 and point[an]< 270:
            x = [0.0, -15.0, -30.0]
            y = get_y(x, point[an])
            if point[p]<point[val] and an == p:
                z = get_y(x, 270)
                response.append((x,y,z))
            elif an==p:
                z = get_y(x, 180)
                response.append((x,z,y))
            else:
                response.append((x,old_p,y))
            if an == p:
                old_p = z
        elif point[an] > 270 and point[an]< 360:
            x = [0.0, 15.0, 30.0]
            y = get_y(x, point[an])
            if point[p]<point[val] and an == p:
                z = get_y(x, 360)
                response.append((x,y,z))
            elif an==p:
                z = get_y(x, 270)
                response.append((x,z,y))
            else:
                response.append((x,old_p,y))
            if an == p:
                old_p = z
    return response,[],[]

def get_point(p, point):
    if p+1 < len(point):
        val  = p+1
    else:
        val = 0
    x = []
    if ((point[p]> 0 and point[p] < 90) or (point[p] < 360 and point[p] > 270)) and ((point[val]> 0 and point[val] < 90) or (point[val] < 360 and point[val] > 270)):
        x = [0, 15, 30]
    elif ((point[p]> 90 and point[p] < 180) or (point[p] < 270 and point[p] > 180)) and ((point[val]> 90 and point[val] < 180) or (point[val] < 270 and point[val] > 180)):
        x = [0, -15, -30]
    else:
        return get_exceptions(p,val,point)
        
    if x: 
        y = get_y(x, point[p])
        z = get_y(x, point[val])
    return x, y, z
    

def plot_g(x,y,xlim,ylim,filename=None,point=None,title=None,c=False,centroid=False):
    """ Plot the points using matplotlib.
        params: x = x points
                y = y points
                xlim = (x_max,x_min) xrange limits for plots
                ylim = (y_max,y_min) yrange limits for plots
                c = colors if avaliable else False
                centroid = centre point (lidar position)
        return: plot is plotted
    """
    fig = plt.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
    if title:
        fig.suptitle(title, fontsize=20)
    plt.xlabel('x axis', fontsize=18)
    plt.ylabel('z axis', fontsize=16)
    ax = fig.gca()
    if c:
        plt.scatter(x, y, s=4, c=c)
    else:
        plt.scatter(x, y, s=4)
    if centroid:
        plt.scatter(0, 0, s=400, c="red")
    plt.grid()
    

    if point:
        for p in range(len(point)):
            endy = 40 * math.sin(math.radians(point[p]))
            endx = 40 * math.cos(math.radians(point[p]))
            plt.plot([0, endx], [0, endy])
            if p%2!=0:
                x_point,y_point,z_point = get_point(p,point)
                if x_point and isinstance(x_point, list) and isinstance(x_point[0], tuple):
                    for xi,yi,zi in x_point:
                        plt.fill_between(xi, yi , zi, color='grey', alpha='0.8') 
                elif x_point and y_point and z_point:
                    plt.fill_between(x_point, y_point , z_point, color='grey', alpha='0.8')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if filename:
        fig.savefig(filename)
    else:
        plt.show()


def get_data(object_type,object,num_scenes):
    """ Load the data from dataset and apply ground removal and outlinear removal.
        params: object_type = train or test 
                object =  Name of the object
        return: dataframe of the data
    """
    dataset_dir = "../dataset/train"
    # Create the outliers
    get_outliers(dataset_dir)
    # Creating the path for the object
    object_path = "../{}/{}/in.csv".format("dataset/{}".format(object_type), object)
    # read in the object of choice
    dataframes = read_file(object_path, num_scenes)
    # remove the outliers
    no_outliers = remove_outliers(dataframes, num_scenes, '../ssh-kd/data/outliers.pkl')
    # get the object points
    return no_outliers


# calculating degree
calDegrees=lambda x : round(math.degrees(math.atan(x)), 1)
vfunc = np.vectorize(calDegrees)
def tang(x,y):
    """ Get Tan inverse of y/x to get the angle."""
    tag= np.divide(y, x)
    return tag


def get_degrees(c):
    """ Get the degree for the data based upon cordinate plane.
    Tan behaviour in cordinate system.
    1st cord: theta
    2nd cord: 180+theta
    3rd cord: 180+theta
    4th cord: 360-theta
    return: degree of dataframe rows
    """
    degrees =  vfunc(tang(c['X'],c['Z']))
    if (c['X']<0 and c['Z']>0):
        degrees = 180+degrees
    if (c['X']<0 and c['Z']<0):
        degrees = 180+degrees
    if (c['X']>0 and c['Z']<0):
        degrees = 360+degrees
    return degrees


def angle_of_elevation(x,y,z):
    """Get Tan inverse of y/sqrt(x^2+z^2) to get the angle"""
    den = math.sqrt(x**2+z**2)
    etan= np.divide(y,den)
    return etan 


def get_phi(c):
    """ Get the degree for the data based upon cordinate plane.
    return: degree of dataframe rows
    """
    phi =  vfunc(angle_of_elevation(c['X'],c['Y'],c['Z']))
    phi =  90-phi
    return phi


def get_r(x,y,z):
    """Get density r"""
    den = x**2+z**2
    return den 

def get_den(c):
    r = get_r(c['X'],c['Y'],c['Z'])
    return r


def stock_sectors(degr, stock,interval = 1.1):
    """ Split the degree in sectors based upon on the gap.
    params: degr=sorted list of unqiue degrees
            stock=List which contains the ranges of degree of a sector
            interval=step range of degree for look up 
    return stock=list contain the ranges of degree of a sector
    """
    sector= [degr[0]]
    for i in range(1, len(degr)):
        if degr[i]-degr[i-1]>interval:
            stock.append(sector)
            sector = []
        sector.append(degr[i])
    stock.append(sector)
    return stock


for dir_ in os.listdir('../dataset/test'): 
    # Experiment with Test data 
    if dir_ == 'Set2':
        num_scenes = 500
    else:
        num_scenes = 50
    no_outliers = get_data("test", dir_, num_scenes=num_scenes)
    # Get the data for a scene - 6
    for each_scn in range(len(no_outliers)):
        scene_df = helper_object_points(no_outliers[each_scn], 4)
        try:
            os.makedirs('../sectors/'+dir_+'/')
        except:
            pass

        plot_g(scene_df['X'],scene_df['Z'],(-25,25),(-25,25),title='{} Scene {}'.format(dir_ ,each_scn),filename='../sectors/'+dir_+'/'+str(each_scn)+'.png',centroid=True)

        # Using Dataframe apply, a func is run on dataframe, pass the datafram to get_degrees
        scene_df['angles'] = scene_df.apply(get_degrees,axis=1)
#         scene_df['phi'] = scene_df.apply(get_phi,axis=1)
#         scene_df['r'] = scene_df.apply(get_den,axis=1)
        try:
            os.makedirs('../sector/'+dir_+'/')
        except:
            pass

        # convert the numpy array in list
        deg = [float(d) for d in scene_df['angles']]
        s = []
        # apply the sorting
        sorted_degrees = sorted(list(set(deg)))
        # apply the stock_sector for breaking the sectors
        s = stock_sectors(sorted_degrees,s)
        print("Number of sectors:",len(s))

        point = []
        for i, each_s in enumerate(s):
            print(max(each_s), min(each_s))
            if max(each_s) == min(each_s):
                continue
            point.append(min(each_s))
            point.append(max(each_s))

        plot_g(scene_df['X'],scene_df['Z'],(-25,25),(-25,25),title='{} Scene {}'.format(dir_ ,each_scn),filename='../sector/'+dir_+'/'+str(each_scn)+'.png', point=point,centroid=True)
