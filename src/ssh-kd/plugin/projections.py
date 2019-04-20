def normalize(df):
    df['_x'] = (df['_x']-df['_x'].min())/(df['_x'].max()-df['_x'].min())
    df['_y'] = (df['_y']-df['_y'].min())/(df['_y'].max()-df['_y'].min())
    return df


def prespective_project(df, d, view):
    if view==2:
        df['_x'] = ((df['X']/abs(df['Z']))*d)
        df['_y'] = ((df['Y']/abs(df['Z']))*d)

    elif view==2:
        df['_x'] = ((df['Z']/abs(df['X']))*d)
        df['_y'] = ((df['Y']/abs(df['X']))*d)
    
    return df


def cabin_projection(df, alpha):
    df['_x'] = (df['X'] + (0.2 * df['Z']) * np.cos(np.deg2rad(alpha)))
    df['_y'] = (df['Y'] + (0.2 * df['Z']) * np.sin(np.deg2rad(alpha)))
    return df
