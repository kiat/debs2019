import numpy as np
from scipy import stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_different_sectors(temp, threshold=0.7):
    """ 
        Divides the scene into sectors. 
    
        In a given 3D scene, from the top view, this functions divides the scene into object containing sectors and empty sectors 
    
        Parameters: 
        temp (numpy.ndarray): The input is a numpy array with X,Y,Z,radius values 
    
        Returns: 
        list_of_sectors (list): list of sectors, where each sector is a numpy.ndarray with X,Y,Z,radius,angles, top_radius values
  
    """
    # Calculate the angles using the np.arctan2
    angles = np.round(np.arctan2(temp[:,2],temp[:,0])*180/np.pi,1).astype(np.float)
    angles[angles<0] = angles[angles<0]+360

    # calculate the top_radius(sqrt(x**2+z**2))
    top_rad = np.round(np.sqrt(temp[:,0]**2+temp[:,2]**2),1)

    # appending angles and the top radius to the input
    temp = np.hstack([temp, angles.reshape(-1,1), top_rad.reshape(-1,1)])

    # Calculating the unique angles
    unique_angles = sorted(list(np.unique(angles)))
    indexes_to_split = list(np.where(np.diff(unique_angles)>=threshold)[0]+1)
    start=0

    if len(unique_angles)==0:
        angle_ranges = [unique_angles]
    else:
        angle_ranges = []

    for i in indexes_to_split:
        angle_ranges.append(unique_angles[start:i])
        start = i
    
    list_of_sectors = []
    
    for j in angle_ranges:
        max_angle = max(j)
        min_angle = min(j)
        bool_vec = (angles>=min_angle) & (angles<=max_angle)
        if np.sum(bool_vec)>10:
            list_of_sectors.append(temp[bool_vec])
    
    return list_of_sectors


def get_valid_density(list_of_valid_sec):
    """ 
        Divides the Sector into mini areas of objects. 
    
        In a given sectors of a 3D scene, from the top view, this functions divides the sectors into object containing area and empty areas 
    
        Parameters: 
        list_of_valid_sec (list of numpy.ndarray): The input is a list of numpy array with X,Y,Z,radius,angles,top_radius values 
    
        Returns: 
        list_of_valid_density (list): list of objects, where each object is a numpy.ndarray with X,Y,Z,radius,angles, top_radius values
  
    """

    # Return value
    list_of_valid_density = []


    for temp in list_of_valid_sec:
        # Initializing the kernel
        try:
            kernel = stats.gaussian_kde(temp[:,5],bw_method=0.05)
        except:
            print(temp[:,5])
            exit(-1)

        # Evaluating the values
        to_plot2 = kernel.evaluate(np.linspace(-20,180,500))

        # Threshold ==0.001
        bool_vec = [~(to_plot2<=0.001)]

        # Selecting valid values wrt threshold
        to_plot = to_plot2[bool_vec]
        x_val = np.linspace(-20,180,500)[bool_vec]

        # Selecting the boundary points
        req_indexes = np.where((np.diff(x_val)<=0.5)==False)[0]+1
        markers_on = x_val[req_indexes].round(0).astype(int).tolist()
        markers_on = [0]+markers_on+[180]

        # Calculate the dense indexs
        to_dense = np.split(to_plot,req_indexes)
        try:
            max_dense = [np.max(j) for j in to_dense]
        except:
            list_of_valid_density.append(temp)
            continue

        # Selecting the valid objects
        for i in range(len(markers_on)-1):
            if max_dense[i]>=0.01:
                temp1 = temp[ (temp[:,5]>=markers_on[i]) & (temp[:,5]<=markers_on[i+1]) ]
                if len(temp1)>=15:
                    list_of_valid_density.append(temp1)
    
    return list_of_valid_density

def prep_obj_data(temp):
    """
        Takes in the data frame and return the list of the valid objects
        
        Parameters: 
        temp (numpy.ndarray): The input is a list of numpy array with X,Y,Z,radius 
    
        Returns: 
        list_of_valid_density (list): list of objects, where each object is a numpy.ndarray with X,Y,Z,radius,angles, top_radius values 
    """
    # Finding the list of sectors 
    list_of_sectors = get_different_sectors(temp)

    # Finding the list of objects
    list_of_valid_density = get_valid_density(list_of_sectors)

    return list_of_valid_density