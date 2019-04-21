import pandas as pd

def min_max_normalize(x,y):
    x = (x-x.min())/(x.max()-x.min())
    y = (y-y.min())/(y.max()-y.min())
    return 7*x,3*y

def standard_normalization(x,y):
    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()

    return x,y

def prespective_project(x,y,z, d, view):
    
    if view==2:
        X = ((x/abs(z))*d)
        Y = ((y/abs(z))*d)
    
    elif view==3:
        X = ((z/abs(x))*d)
        Y = ((y/abs(x))*d)
    
    X,Y = standard_normalization(X, Y)
    
    return pd.DataFrame({'X':X,'Y':Y})


def cabin_projection(df, alpha):
    df['_x'] = (df['X'] + (0.2 * df['Z']) * np.cos(np.deg2rad(alpha)))
    df['_y'] = (df['Y'] + (0.2 * df['Z']) * np.sin(np.deg2rad(alpha)))
    return df
