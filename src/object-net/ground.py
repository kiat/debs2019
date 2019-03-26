from data import remove_outliers, visualize, visualize_base
import numpy as np
import pandas as pd
from datetime import datetime

def main():
    # Take the outliers
    # outliers , others = remove_outliers("/home/samba693/DataChallenge/debs2019_initial_dataset/Atm/in.csv")

    # # get the min and max values for each outlier range 
    # for i in outliers:
    #     i['radiusSquare'] = i['X']**2+i['Y']**2+i['Z']**2
    #     i['radius'] = np.sqrt(i['radiusSquare']).round(1)
    #     i = i[i['radius']>0]
    #     i['max'] = i.groupby(['lz'])['radius'].transform('max')
    #     i['min'] = i.groupby(['lz'])['radius'].transform('min')
    #     i = i[['lz','max','min']]
    #     i.drop_duplicates(subset=['lz','max','min'],inplace=True)
    
    # # Save the data frame
    # i.to_pickle("./outliers.pkl")

    outliers = pd.read_pickle("./outliers.pkl")
    path = "/home/samba693/DataChallenge/debs2019_initial_dataset/Pedestrian/in.csv"
    dataframes = object_points(path, outliers)
    visualize(dataframes,2)

    
def object_points(path, outliers):
    dataframes = []
    for i in range(50):
        cur = datetime.now()
        df = pd.read_csv(path, usecols=[1,2,3,4], skiprows=i*72000, nrows=72000, names=["lz","X","Y","Z"])
        df['radiusSquare'] = df['X']*df['X']+df['Y']*df['Y']+df['Z']*df['Z']
        df['radius'] = np.sqrt(df['radiusSquare']).round(1)
        temp_out = pd.DataFrame()
        for j in range(64):
            max_rad = outliers[outliers['lz']==j]['max'].tolist()[0]
            min_rad = outliers[outliers['lz']==j]['min'].tolist()[0]
            dummy_df = df[df['lz']==j]
            temp_out = temp_out.append(dummy_df[~((dummy_df['radius']<=max_rad) & (dummy_df['radius']>=min_rad))])
        temp_out.drop(temp_out[temp_out['radius']==0].index, inplace = True)
        dataframes.append(temp_out)
        print("Stop for {} images = {}".format(i,datetime.now() - cur))
    return dataframes

if __name__ == "__main__":
    main()