import os
import random

import numpy as np

def set_seed(seed=666):
    '''
    Set random seed.
    
    Args:
        seed (int): number for random seed 
    '''

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
