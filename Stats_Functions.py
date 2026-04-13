import math
import numpy as np
def standardNormal(z):
    a1 = 0.196854
    a2 = 0.115194
    a3 = 0.000344
    a4 = 0.019527
    if z >= 0:
        return 1 - (1/2) * (1 + a1 * z + a2 * z * z + a3 * z * z * z + a4 * z * z * z *z) ** -4
    else:
        return 1 - standardNormal(-z)

def standardNormalArray(nums):
    outputList = []
    for n in nums.tolist():
        outputList.append(standardNormal(n))
    return np.array(outputList)

def inverseStandardNormal(p):
    a0 = 2.30753
    a1 = 0.27061
    b1 = 0.99229
    b2 = 0.04481
    if p < 0 or p > 1:
        return None
    if p >= 0.5:
        t = (-2 * math.log(1 - p)) ** (1/2)
        num = a0 + a1 * t
        den = 1 + b1 * t + b2 * t * t
        return t - num/den
    else:
        return -1 * inverseStandardNormal(1-p)

def normal(x, mu, sigma):
    return standardNormal((x - mu) / sigma)

def inverseNormal(p, mu, sigma):
    return sigma * inverseStandardNormal(p) + mu

def sampleMean(list):
    sum = 0.0
    count = 0.0
    for l in list:
        sum += l
        count += 1
    return sum / count

def sampleVariance(list, unbiased):
    m = sampleMean(list)
    sum = 0.0
    count = 0.0
    for l in list:
        sum += (l - m)**2
        count += 1
    if unbiased:
        return sum / (count - 1)
    else:
        return sum / count

def standardPlottingPositions(len):
    output = []
    for i in range(1,len+1):
        output.append(float(i) / len)
    return output

def WeibullPlottingPositions(len):
    output = []
    for i in range(1,len+1):
        output.append(i / (len + 1.0))
    return output

def metaGaussianPlottingPositions(len):
    output = []
    if len < 3:
        tn = 3.0193 * (len ** -1.1018) + 1
    elif len < 6:
        tn = 2.4035 * (len ** -0.9096) + 1
    elif len < 11:
        tn = 2.1408 * (len ** -0.8423) + 1
    elif len < 20001:
        tn = 1.9574 * (len ** -0.8039) + 1
    else:
        tn = 1
    for n in range(1,len+ 1):
        inner = (len - n + 1) / n
        inner = inner ** tn
        output.append((inner + 1) ** -1)
    return output

def LSregression(v, u, a, b):
    N = len(v)
    v = np.array(v)
    u = np.array(u)
    vBar = np.sum(v) / N
    uBar = np.sum(u) / N
    bHat = 0
    aHat = 0
    if a and b:
        bHat = (np.sum(v * u) - N * vBar * uBar) / (np.sum(u ** 2) - N * (uBar ** 2))
        aHat = vBar - bHat * uBar
    if b and not a:
        bHat = np.sum(v * u) / np.sum(u ** 2)
    if a and not b:
        aHat = vBar - uBar
    return aHat, bHat

def calcMAD(p, estimates):
    curMax = 0
    for i in range(len(p)):
        tempMax = abs(p[i] - estimates[i])
        if tempMax > curMax:
            curMax = tempMax
    return curMax

def calcKStest(estimates):
    N = len(estimates)
    Dlmax = 0
    DAmax = 0
    for i in range(N):
        Dltemp = abs((i-1)/N - estimates[i])
        DAtemp = abs(i/N - estimates[i])
        if Dltemp > Dlmax:
            Dlmax = Dltemp
        if DAtemp > DAmax:
            DAmax = DAtemp
    return Dlmax, DAmax, max(Dlmax, DAmax)

def normalDensity(nums,mu, sigma):
    coef = 1 / (sigma * math.sqrt(2 * math.pi))
    output = coef * np.exp( (-1/2) * np.power((nums - mu)/sigma,2))
    return output

def getDownloadsTab():
    return "C:/Users/ucg8nb\Downloads"

def getCorrelation(nqtX, nqtY):
    nqtX = np.array(nqtX)
    nqtY = np.array(nqtY)
    bothBar = np.mean(nqtX * nqtY)
    xBar = np.mean(nqtX)
    yBar = np.mean(nqtY)
    xSTD = np.std(nqtX, ddof = 1)
    ySTD = np.std(nqtY, ddof = 1)
    return (bothBar  - xBar * yBar) / (xSTD * ySTD)

def logisticDF(nums, alpha, beta):
    return np.power(1 + np.exp(-1 * (nums - beta) / alpha), -1)

def logisticdf(nums, alpha, beta):
    return (1 / alpha) * np.exp(-1 * (nums - beta) / alpha) * np.power(1 + np.exp(-1 * (nums - beta) / alpha), -2)

def laplaceDF(nums, alpha, beta):
    numbers = nums.tolist()
    outputList = []
    for n in numbers:
        if n <= beta:
            outputList.append((1/2) * np.exp((n - beta) / alpha))
        if n > beta:
            outputList.append(1 - (1/2) * np.exp(-1 * (n - beta) / alpha))
    return np.array(outputList)

def laplacedf(nums, alpha, beta):
    return (1 / (2 * alpha)) * np.exp(-1 * np.abs(nums - beta) / alpha)

def gumbelDF(nums, alpha, beta):
    return np.exp(-1 * np.exp(-1 * (nums - beta) / alpha ))

def gumbeldf(nums, alpha, beta):
    return (1/ alpha) * np.exp(-1 * (nums - beta) / alpha) * np.exp(-1 * np.exp(-1 * (nums - beta) / alpha))

def reflectedGumbelDF(nums, alpha, beta):
    return 1 - np.exp(-1 * np.exp((nums - beta) / alpha ))

def reflectedGumbeldf(nums, alpha, beta):
    return (1 / alpha) * np.exp((nums - beta) / alpha) * np.exp(-1 * np.exp((nums - beta) / alpha ))

def exponentialDF(nums, alpha, eta):
    return 1 - np.exp(-1 * (nums - eta) / alpha)

def exponentialdf(nums, alpha, eta):
    return (1 / alpha) * np.exp(-1 * (nums - eta) / alpha)

def weibullDF(nums, alpha, beta, eta):
    return 1 - np.exp(-1 * np.power((nums - eta) / alpha, beta))

def weibulldf(nums, alpha, beta, eta):
    return (beta / alpha) * np.power((nums - eta) / alpha, beta - 1) * np.exp(-1 * np.power((nums - eta) / alpha, beta))

def invertedWeibullDF(nums, alpha, beta, eta):
    return np.exp(-1 * np.power(alpha / (nums - eta) ,beta))

def invertedWeibulldf(nums, alpha, beta, eta):
    return (beta / alpha) * np.exp(alpha / (nums - eta), beta + 1) * np.exp(-1 * np.power(alpha / (nums - eta), beta))

def logWeibullDF(nums, alpha, beta, eta):
    return 1 - np.exp(-1 * np.power(np.log(nums - eta + 1) / alpha, beta))

def logWeibulldf(nums, alpha, beta, eta):
    return (beta / (alpha * np.log(nums - eta + 1))) * np.power(np.log(nums - eta + 1) / alpha, beta -1) * np.exp(-1 * np.power(np.log(nums - eta + 1) / alpha, beta))

def logLogisticDF(nums, alpha, beta, eta):
    return np.power(1 + np.power((nums - eta) / alpha, -1 * beta), -1)

def logLogisticdf(nums, alpha, beta ,eta):
    return (beta / alpha) * np.power((nums - eta) / alpha, -1 * beta - 1) * np.power(1 + np.power((nums - eta) / alpha, -1 * beta), -2)

def power1DF(nums, beta, etaL, etaU):
    return np.power((nums - etaL) / (etaU - etaL), beta)

def power1df(nums, beta, etaL, etaU):
    return (beta / (etaU - etaL)) * np.power((nums - etaL) / (etaU - etaL), beta - 1)

def power2DF(nums, beta, etaL, etaU):
    return 1 - np.power((etaU - nums) / (etaU - etaL), beta)

def power2df(nums, beta, etaL, etaU):
    return (beta / (etaU - etaL)) * np.power((etaU - nums) /  (etaU - etaL), beta - 1)

def normalDF(nums, mu, sigma):
    outputList = []
    for n in nums.tolist():
        outputList.append(normal(n, mu, sigma))
    return np.array(outputList)

def normaldf(nums, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp((-1 / 2) * np.power((nums - mu)/sigma, 2))

def logNormalDF(nums, mu, sigma, eta):
    transformedNums = (np.log(nums - eta) - mu) / sigma
    return normalDF(transformedNums, mu, sigma)

def logNormaldf(nums, mu, sigma, eta):
    coef = 1 / ((nums - eta) * sigma * math.sqrt(2 * math.pi))
    transformedNums = (np.log(nums - eta) - mu) / sigma
    return coef * np.exp((-1 / 2) * np.power(transformedNums, 2))

def logRatioDF(nums, type, alpha, beta, etaL, etaU):
    transformedNums = np.log((nums - etaL) / (etaU - nums))
    if type == "LG":
        return logisticDF(transformedNums, alpha, beta)
    if type == "LP":
        return laplaceDF(transformedNums, alpha, beta)
    if type == "GB":
        return gumbelDF(transformedNums, alpha, beta)
    if type == "RG":
        return reflectedGumbelDF(transformedNums, alpha, beta)
    if type == 'NM':
        return normalDF(transformedNums, alpha, beta)

def logRatiodf(nums, type, alpha, beta, etaL, etaU):
    transformedNums = np.log((nums - etaL) / (etaU - nums))
    coef = (etaU - etaL) / ((nums - etaL) * (etaU - nums))
    if type == "LG":
        return coef * logisticdf(transformedNums, alpha, beta)
    if type == "LP":
        return coef * laplacedf(transformedNums, alpha, beta)
    if type == "GB":
        return coef * gumbeldf(transformedNums, alpha, beta)
    if type == "RG":
        return coef * reflectedGumbeldf(transformedNums, alpha, beta)
    if type == "NM":
        return coef * normaldf(transformedNums, alpha, beta)

def empiricalNQT(sample):
    plotPos = metaGaussianPlottingPositions(len(sample))
    sortedSample = sorted(sample)
    outputList = []
    for n in sample:
        index = sortedSample.index(n)
        outputList.append(plotPos[index])
    return outputList

def inverseStandardNormalArray(nums):
    outputList = []
    for p in nums.tolist():
        outputList.append(inverseStandardNormal(p))
    return np.array(outputList)

def xi(gamma, z1, z2):
    return (1 / math.sqrt(1 - gamma)) * np.exp(((-1 * gamma) / (2 * (1 - gamma**2))) * (gamma * np.power(z1, 2) + z1 * z2 + gamma * np.power(z2, 2)) )

def pix(g, lam, den1, den0):
    prior = (1 - g) /g
    return np.power(1 +  prior * lam * den0 / den1, -1)

def inverseLG(p, alpha, beta):
    return beta + alpha * np.log(p / (1 - p))

def inverseLP(p, alpha, beta):
    listP = p.to_list()
    outputList = []
    for realP in listP:
        if realP <= 0.5:
            outputList.append(beta + alpha * np.log(2 * realP))
        else:
            outputList.append(beta - alpha * np.log(2 * (1 * realP)))
    return np.array(outputList)

def inverseGB(p, alpha, beta):
    return beta - alpha * np.log(-1 * np.log(p))

def inverseRG(p, alpha, beta):
    return beta + alpha * np.log(-1 * np.log(1 - p))

def inverseEX(p, alpha, eta):
    return -1 * alpha * np.log(1 - p) + eta

def inverseWB(p, alpha, beta, eta):
    return alpha * np.power(-1 * np.log(1 - p), 1/ beta) + eta

def inverseIW(p, alpha, beta ,eta):
    return alpha * np.power(-1 * np.log(p), -1 / beta) + eta

def inverseLW(p, alpha, beta, eta):
    return np.exp(alpha * np.power(-1 * np.log(1 - p), 1 / beta)) + eta - 1

def inverseLL(p, alpha, beta, eta):
    return alpha * np.power(p / (1 - p), 1/beta) + eta

def inverseP1(p, beta, etaL, etaU):
    return (etaU - etaL) * np.power(p, 1/beta) + etaL

def inverseP2(p, beta, etaL, etaU):
    return etaU - (etaU - etaL) * np.power(1 - p, 1/beta)

def inverseLR(p, type, alpha, beta, etaL, etaU):
    if type == "LG":
        inverseP = inverseLG(p, alpha, beta)
    elif type == "LP":
        inverseP = inverseLP(p, alpha, beta)
    elif type == "GB":
        inverseP = inverseGB(p, alpha, beta)
    else:
        inverseP = inverseRG(p, alpha, beta)
    return etaU - ((etaU - etaL) / (np.exp(inverseP) + 1))

