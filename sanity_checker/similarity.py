
from sklearn.metrics import mean_squared_error
import numpy as np
def mse(A, B):
    return (np.square(A - B)).mean(axis=None)

from scipy.stats import spearmanr 
def spearman_rank(A, B):
    result = 0.0
    for i in range(len(A)):
        result += spearmanr(A[i], B[i], axis=None)[0]
    return result / len(A)