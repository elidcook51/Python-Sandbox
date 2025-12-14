import FinalExamHelper as help
from scipy.stats import norm
import numpy as np
from sympy import Matrix


matrix = np.array([
    [0.0225, 0.018, 0.05],
    [0.018, 0.09, 0.1]
])

M = Matrix(matrix)
rref_M, pivot_cols = M.rref()

rref = np.array(rref_M)

lastCol = rref[:,2]

totVal = np.sum(lastCol)

normalized = lastCol / totVal

wa = normalized[0]
wb = normalized[1]

print(lastCol)
print(normalized)

print(wa * 0.1 + wb * 0.15)
print(wa**2 * 0.15 ** 2 + wb ** 2 + 0.3 ** 2 + wa * wb * 0.018)


print(np.array(rref_M))




# sigmas =np.array([.2, .24, .3, .4])
# # sigmas = np.arange(0, 1, 0.001)
# impliedSigmas = [0.4, 0.3, 0.2, 0.24]
# S = 125
# r = 0.05
# T = 1

# Kvals = [50, 75, 125, 150]
# prices = [77.51, 53.99, 13.06, 5.82]

# sigma3 = 0.2
# sigma4 = 0.24

# K3 = 125
# K4 = 150

# delta3=  norm.cdf(help.d1(S, K3, r, T, sigma3))
# delta4 = norm.cdf(help.d1(S, K4, r, T, sigma4))

# print(delta3, delta4)

# print(help.bsCall(100, K3, r, T, sigma3))
# print(norm.cdf(help.d1(100, K3, r, T, sigma3)))