import numpy as np

def changeInC(A, b, c, index, basic):
    if index in basic:
        totIndex = list(range(A.shape[1]))
        basicNum = basic.index(index)
        nonBasic = [ind for ind in totIndex if ind not in basic]
        minMax = [-np.inf, np.inf]
        for ind in nonBasic:
            cp = c[ind]
            basicNonIndex = [ind for ind in basic if ind != index]
            costNoIndex = c[basicNonIndex]
            invAb = np.linalg.inv(A[:,basic])
            Ap = A[:,ind]
            rhoMat = invAb @ Ap
            basicNonBasicNum = [basic.index(ind) for ind in basic if ind != index]
            noIndexRho = rhoMat[basicNonBasicNum]
            rho = rhoMat[basicNum]
            minOrMax = (cp - np.dot(costNoIndex, noIndexRho)) / rho
            if rho > 0:
                if minOrMax < minMax[1]:
                    minMax[1] = minOrMax
                print(minOrMax, 'Max', ind)
            else:
                if minOrMax > minMax[0]:
                    minMax[0] = minOrMax
                print(minOrMax, 'Min', ind)
        return minMax
    else:
        invAb = np.linalg.inv(A[:,basic])
        Ap = A[:, index]
        cp = c[index]
        cbasic = c[basic]
        return cp - cbasic @ invAb @ Ap

A = np.array([
    [1, 0, 5, 2],
    [0, 2, 1, 3]
])

c = np.array([0, 0, 2, 5])

b = np.array([[30],[60]])

print(changeInC(A, b, c, 2, [0,1]))