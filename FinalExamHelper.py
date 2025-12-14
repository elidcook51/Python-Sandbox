import numpy as np
from sympy import Matrix
from scipy.stats import norm

#To find RREf using sympy, convert np matrix to sympy matrix using M = Matrix(np matrix)
#Then call rref_matrix, pivot_cols = M.rref()
#Can convert back with np.array(M)



def d1(S, K, r, T, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma) - sigma * np.sqrt(T)

def bsCall(S, K, r, T, sigma):
    d1val = d1(S, K, r, T, sigma)
    d2val = d2(S, K, r, T, sigma)

    
    Nd1 = norm.cdf(d1val)
    Nd2 = norm.cdf(d2val)

    output = S * Nd1 - K * np.exp(-1 * r * T) * Nd2
    return output

def bsPut(S, K, r, T, sigma):
    d1val = d1(S, K, r, T, sigma)
    d2val = d2(S, K, r, T, sigma)

    Nd1 = norm.cdf(-1 * d1val)
    Nd2 = norm.cdf(-1 * d2val)

    return K * np.exp(-1 * r *  T) * Nd2 - S * Nd1

def getSigmaCall(S, K, r, T, cost, sigmas):
    output = []
    for s in sigmas.tolist():
        output.append(bsCall(S, K, r, T, s))
    bsVals = np.array(output)
    bsVals = np.abs(bsVals - cost)
    return sigmas[bsVals.tolist().index(np.min(bsVals))]




