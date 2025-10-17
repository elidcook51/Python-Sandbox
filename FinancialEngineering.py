import numpy as np

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
g=400
d = 1 / 1.1

Ki = [213.81, 211.45, 208.17, 203.58, 197.13, 187.96, 174.79, 155.47, 126.28, 80.00, 0]
Knew = [220] * 11
xi = [50000]
zi = []
for i in range(len(Knew) - 1):
    K = Knew[i+ 1]
    x=  xi[i]
    z = (g - d * K) * x / 1000
    zi.append(z)
    newX = x - z
    xi.append(newX)
zi.append(0)
year = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
colNames = ['Year', '$K_{i+1}$', '$x_i$', '$z_i$']
printInLatexTable([year,Knew, xi, zi], colNames)
