
import numpy as np


def Linear(X):
    """Linear mean function."""
    H = np.hstack( [np.ones((X.shape[0],1)) , X] )
    return H


def Constant(X):
    """Constant mean function."""
    H = np.ones((X.shape[0],1))
    return H


