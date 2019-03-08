# Debs 2019

# Packages
import pandas
import numpy as np
import os
    
# Visualization 
from plotly.offline import plot, iplot
import plotly.graph_objs as go

# Data Grouping
from sklearn.cluster import KMeans

# Dataset
for filen in os.listdir('dataset/'):
    try: 
        print("Grouping {}".format(filen))
        directory = 'dataset/{}'.format(filen)
        in_path = '{}/in.csv'.format(directory)

        out_path = '{}/out.csv'.format(directory)
        out_file = open(out_path,'r').readlines()
        output_con = []
        _totalsum = 0
        for line in out_file:
            line_sum = sum([int(con) for con in line.strip().split(',')[1:] if len(con)<3])
            _totalsum+=line_sum
            output_con.append((int(line[0:1]),line_sum))

        x,y,z=[],[],[]
        # Reading in the data for one scene at a time
        giant_df = pandas.DataFrame(columns=['timestamp','id','x','y','z','r_sq','r','freq'])
        for j in range(len(output_con)):
            df = pandas.read_csv(in_path, skiprows=j*72000, nrows=72000, names=["timestamp", "id", "x", "y", "z"])
            df['r_sq'] = df['x']**2+df['y']**2+df['z']**2
            df['r'] = np.sqrt(df['r_sq']).round(1)
            df['freq'] = df.groupby(['id','r'])['r'].transform('count')
            for i in range(64):
                maxfreq = df[(df['id']==i) & (df['r']!=0)]['freq'].max()
                while maxfreq>100:
                    df.drop(df[(df['id']==i) & (df['freq']==maxfreq)].index,inplace=True)
                    maxfreq = df[(df['id']==i) & (df['r']!=0)]['freq'].max()
                df.drop(df[(df['id']==i) & (df['r']==0)].index,inplace=True)
            giant_df = giant_df.append(df)
            # Creating the tuples
            x = tuple(df['x'].tolist())
            y = tuple(df['y'].tolist())
            z = tuple(df['z'].tolist())
            
            kmeans_model = KMeans(n_clusters=output_con[j][1]+2, random_state=1).fit(df.iloc[:, 2:5])
            labels = kmeans_model.labels_

            trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=labels, size=2, opacity=0.4))
            data = [trace]
            print("grouping completed for {}".format(j))
            
            if not os.path.exists('visuals/'+filen):
                os.makedirs('visuals/'+filen)
            plot(data,filename='visuals/'+filen+'/in_{}.html'.format(j),auto_open=False,show_link=False)

        # Creating the tuples
        xs = tuple(giant_df['x'].tolist())
        ys = tuple(giant_df['y'].tolist())
        zs = tuple(giant_df['z'].tolist())

        kmeans_model = KMeans(n_clusters=_totalsum+2, random_state=1).fit(df.iloc[:, 2:5])
        labels = kmeans_model.labels_

        trace = go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(color=labels, size=2, opacity=0.4))
        data = [trace]
        plot(data,filename='visuals/{}/gaint_model.html'.format(filen),auto_open=False,show_link=False)
        print("Grouped {}".format(filen))
    except:
        pass
