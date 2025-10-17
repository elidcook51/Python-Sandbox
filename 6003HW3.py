import numpy as np

def findDirection(A, b, p):
    A_b = A[:, b]
    A_p = A[:, p]
    A_inv = np.linalg.inv(A_b)
    return -1 * A_inv @ A_p

def findBasicSol(A, basis, b):
    A_b = A[:,basis]
    return np.linalg.inv(A_b) @ b

def findStepSize(A, basis, p, b):
    d = findDirection(A, basis, p)
    basic = findBasicSol(A, basis, b)
    curMin = np.inf
    for elm in basis:
        x = basic[elm]
        db = d[elm]
        if db < 0:
            value = -1 * x / db
            if value < curMin:
                curMin = value
    return curMin


A = np.array([
    [0, 1, 1, 2, 1],
    [4, 0, 1, 1, 2]
])

b = np.array([
    [1],
    [8]
])

basis = [0,1]
p = 3

print(findDirection(A, basis, p))
print(findStepSize(A, basis, p, b))
print(takeStep(A, basis, p, b))