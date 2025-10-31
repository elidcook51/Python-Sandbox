import numpy as np
import pandas as pd

buyCount = [10000, 7500, 5000]
buyCost = [32125, 25250, 17750]

attendance = [40000, 70000, 100000]
attendPerc = [0.24, 0.5, 0.26]

percentBuy = [0.05, 0.1, 0.15]
percentBuyPerc = [0.3, 0.6, 0.1]

outputDf = pd.DataFrame()

def getProfit(b, a, s, bp):
    return min(b, a * s) * (100 / 12) + max(0, b - (a * s)) * 1.5 - bp

for i in range(3):
    for j in range(3):
        for k in range(3):
            b = buyCount[i]
            a = attendance[j]
            s = percentBuy[k]
            totPerc = percentBuyPerc[k] * attendPerc[j]
            bp = buyCost[i]
            profit = getProfit(b,a,s,bp)
            newRow = {
                '# of Shirts': b,
                'Price of Shirts': bp,
                'Attendance': a,
                'Percent Buy': s,
                'Percent Happening': totPerc,
                'Profit': profit
            }
            outputDf = outputDf._append(newRow, ignore_index= True)
outputDf.to_csv("C:/Users/ucg8nb/Downloads/T Shirt Case.csv")