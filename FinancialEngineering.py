import numpy as np
from scipy.stats import norm
import pandas as pd


def printInLatexTable(listOfLists, colNames):
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|" + "c|" * len(colNames)  + "}")
    print("\\hline")
    print(" & ".join(colNames) + "\\\\")
    print("\\hline")
    for i in range(len(listOfLists[0])):
        row_items = []
        for l in listOfLists:
            toAdd = l[i]
            try:
                toAdd = float(toAdd)
                toAdd = np.round(toAdd, decimals=4)
            except ValueError:
                pass
            row_items.append(str(toAdd))
        outputString = " & ".join(row_items) + " \\\\"
        print(outputString)
    print("\\hline")
    print("\\end{tabular}")
    print('\\end{table}')

def d1(S, K, r, T, sigma):
    return (np.log(S/K) + (r + np.power(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma)- sigma * np.sqrt(T)

def blackscholes(S, K, r, T, sigma):
    d1 = (np.log(S/K) + (r + np.power(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    # Nd1 = np.array([norm.cdf(x) for x in list(d1)])
    # Nd2 = np.array([norm.cdf(x) for x in list(d2)])
    output = S * Nd1 - K * np.exp(-1 * r * T) * Nd2
    return output

sigmas = np.arange(0,1,0.000001)
S = 125
K = 120
r = 0.05
T = 63/252
C = 15.2043
sigma = 0.481911

# specD1 = d1(S, K, r, T, sigma)
# print(specD1)
# print(norm.cdf(specD1))

# print(blackscholes(S, K, r, T, 0.4))
ABCprice = pd.read_excel("C:/Users/ucg8nb/Downloads/ABC.xlsx")
ABC = ABCprice['ABC'].tolist()
ts = ABCprice['t'].tolist()
Tvals = [(63 - t) / 252 for t in ts]
d1s = [d1(stock, K, r, t, sigma) for stock, t in zip(ABC, Tvals)]
delta = [norm.cdf(d1) for d1 in d1s]
callprice = [blackscholes(stock, K, r, t, sigma) for stock, t in zip(ABC, Tvals)]
putprice = [call + K / (1 + r) - stock for (call, stock) in zip(callprice, ABC)]
stockAmount = [delt - 1 for delt in delta]
stockPortfolio = [quantity * stock for (quantity, stock) in zip(stockAmount, ABC)]
portVal = [putprice[0]]
bondPort = [portVal[0] - stockPortfolio[0]]
for i in range(1,len(stockPortfolio)):
    stock = ABC[i]
    prevStock = ABC[i - 1]
    stockPort = stockPortfolio[i-1]
    newStockVal = stockPort * (stock / prevStock)
    newbondPort = bondPort[i-1] * (1 + r/63)
    portVal.append(newStockVal + newbondPort)
    bondPort.append(portVal[i] - stockPortfolio[i])
gains = [val - putprice[0] for val in portVal]


newSheet = pd.DataFrame()
newSheet['t'] = ts
newSheet['ABC'] = ABC
newSheet['Delta'] = delta
newSheet['Call Price'] = callprice
newSheet['Put Price'] = putprice
newSheet['Stock Portfolio'] = stockPortfolio
newSheet['Bond Portfolio'] = bondPort
newSheet['Portfolio Value'] = portVal
newSheet['Gain/Loses'] = gains

newSheet.to_csv("C:/Users/ucg8nb/Downloads/HW3 - Class Problem.csv")



# d1s = np.array([d1(S, K, r, T, sigma) for sigma in sigmas])
# outputs = np.array([blackscholes(S, K, r, T, sigma) for sigma in sigmas])
# distFromC = np.abs(outputs - C)
# index = int(np.argmax(distFromC == np.min(distFromC)))
# print(sigmas[index])
# print(distFromC[index])
# print(outputs[index])
# print(d1s[index])

# print(blackscholes(S, K, r, T, 0.4))
# call1 = blackscholes(S, K, 0.04, T, sigma)
# call2 = blackscholes(S, K, 0.05, T, sigma)
# call3 = blackscholes(S, K, 0.06, T, sigma)

# rs = [0.04, 0.05, 0.06]
# calls = [blackscholes(S, K, R, T, sigma) for R in rs]
# puts = [c + K / (1 + r) - S for c,r in zip(calls, rs)]

# colNames = ['Risk Free Rate ($r$)', 'Call Price', 'Put Price']

# printInLatexTable([rs, calls ,puts], colNames)
# d1 = (np.log(S/K) + (r + np.power(sigmas, 2) / 2) * T) / (sigmas * np.sqrt(T))
# d2 = d1 - sigmas * np.sqrt(7)
# Nd1 = np.array([norm.cdf(x) for x in list(d1)])
# Nd2 = np.array([norm.cdf(x) for x in list(d2)])
# output = S * Nd1 - K * np.exp(-1 * r * T) * Nd2
# distFromC = np.abs(output - C)
# index = int(np.argmax(distFromC == np.min(distFromC)))
# print(distFromC[index])
# print(output[index])
# print(sigmas[index])
# print(d1[index])
# print(Nd1[index])
# print(np.where(distFromC == np.min(distFromC)))


# bonusOptions = np.arange(0, 61000, 10000)
# base = 80000
# salaryOptions = bonusOptions + base
# utility = np.power(salaryOptions, 1/4)
# probs = np.array([1/7] * len(utility))
# probString = ['1/7'] * len(utility)

# colNames = ['Bonus Amount', 'Total Salary', 'Utility', 'Probability']
# printInLatexTable([bonusOptions, salaryOptions, utility, probString], colNames)

# print(np.sum(utility * probs))
# print(np.sum(utility * probs) ** 4)

# termStructure = np.array([0.070, 0.073, 0.077, 0.081, 0.084, 0.088])

# shortRates = [termStructure[0]]
# for i in range(len(termStructure) - 1):
#     ind = i + 1
#     num = np.power(1 + termStructure[ind], ind + 1)
#     den = np.power(1+ termStructure[i], i + 1)
#     shortRates.append(num / den - 1)

# shortRates = np.array(shortRates)
# principal = 10000000

# floatPayment = shortRates * principal
# todaysFloatPayment = []
# for i in range(len(floatPayment)):
#     curPayment = floatPayment[i]
#     curRate = termStructure[i]
#     discountRate = 1 / np.power((1 + curRate), i + 1)
#     todaysFloatPayment.append(curPayment * discountRate)

# years = [1, 2, 3, 4, 5, 6]

# colNames = ['Year', 'Short Rate', 'Floating Payment', "Floating Payment today's money"]
# printInLatexTable([years, shortRates, floatPayment, todaysFloatPayment], colNames)

# print(np.sum(todaysFloatPayment))

# A = np.array([
#     [1.4, 2.0, 1.1],
#     [1.1, 1.0, 1.1],
#     [0.9, 0.7, 1.1]
# ])

# ss1 = np.array([
#     [1],
#     [0],
#     [0]
# ])

# ss2 = np.array([
#     [0],
#     [1],
#     [0]
# ])

# ss3 = np.array([
#     [0],
#     [0],
#     [1]
# ])

# sol1 = np.linalg.inv(A) @ ss1
# print(sol1)

# u = 1.1052
# d = 0.9048
# g=400
# d = 1 / 1.1

# Ki = [213.81, 211.45, 208.17, 203.58, 197.13, 187.96, 174.79, 155.47, 126.28, 80.00, 0]
# Knew = [220] * 11
# xi = [50000]
# zi = []
# for i in range(len(Knew) - 1):
#     K = Knew[i+ 1]
#     x=  xi[i]
#     z = (g - d * K) * x / 1000
#     zi.append(z)
#     newX = x - z
#     xi.append(newX)
# zi.append(0)
# year = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# colNames = ['Year', '$K_{i+1}$', '$x_i$', '$z_i$']
# printInLatexTable([year,Knew, xi, zi], colNames)
