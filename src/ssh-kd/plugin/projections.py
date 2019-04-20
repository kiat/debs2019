def normalize(x,y):
    x = (x-x.min())/(x.max()-x.min())
    y = (y-y.min())/(y.max()-y.min())
    return x,y


def prespective_project(df, d, view):
    if view==2:
        df['_x'] = ((df['X']/abs(df['Z']))*d)
        df['_y'] = ((df['Y']/abs(df['Z']))*d)
    
    elif view==3:
        df['_x'] = ((df['Z']/abs(df['X']))*d)
        df['_y'] = ((df['Y']/abs(df['X']))*d)
    
    df['X'], df['Y'] = normalize(df['_x'], df['_y'])
    
    return df


def cabin_projection(df, alpha):
    df['_x'] = (df['X'] + (0.2 * df['Z']) * np.cos(np.deg2rad(alpha)))
    df['_y'] = (df['Y'] + (0.2 * df['Z']) * np.sin(np.deg2rad(alpha)))
    return df
