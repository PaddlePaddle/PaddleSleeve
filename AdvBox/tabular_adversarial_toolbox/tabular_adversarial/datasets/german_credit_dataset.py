import pandas as pd

def load_gcd(file_path):
    '''
    Load German Credit Dataset from file "german.data".

    Args:

    Returns:
        X: samples
        Y: labels
    '''

    df = pd.read_csv(file_path, header=None, sep=' ')

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    return X, Y
