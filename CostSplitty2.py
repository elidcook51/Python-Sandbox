import pandas as pd
import numpy as np

initialPath = "C:/Users/ucg8nb/Downloads/Cost Splitty Spring Break 2026 (Responses) - Form Responses 1.csv"
transformPath = "C:/Users/ucg8nb/Downloads/Transformed Cost Splitty.csv"
outputPath = "C:/Users/ucg8nb/Downloads/Final Output.csv"


def getPeopleList(df):
    return list(set((df['People?'].dropna().str.split(',').explode().str.strip().str.lstrip().unique()).tolist()))

def getPayments(df, toPay, payer):
    existingPayment = df[df['Need to Pay'] == toPay]
    existingPayment = existingPayment[existingPayment['Pay to'] == payer]
    reversePayment = df[df['Need to Pay'] == payer]
    reversePayment = reversePayment[reversePayment['Pay to'] == toPay]
    return existingPayment, reversePayment

def transformFormResponse(df):
    peopleList = getPeopleList(df)
    outputDf = pd.DataFrame()
    for index, row in df.iterrows():
        startingDict = {name.strip().lstrip(): 0 for name in peopleList}
        users = []
        users = str(row['People?']).split(',')
        for u in users:
            if u.strip().lstrip() in startingDict:
                startingDict[u.strip().lstrip()] = 1
        startingDict['Payer'] = row['Who paid']
        startingDict['Cost'] = row['Cost']
        startingDict['Reason'] = row['Item?']
        outputDf = outputDf._append(startingDict, ignore_index = True)
    firstCols = ['Payer', 'Cost', 'Reason']
    otherCols = [col for col in outputDf.columns if col not in firstCols]
    outputDf = outputDf[firstCols + otherCols]
    return outputDf

def computeCosts(transformedDf, peopleList):
    outputDf = pd.DataFrame(columns = ['Need to Pay', 'Cost', 'Pay to'])
    for index, row in transformedDf.iterrows():
        payer = row['Payer']
        cost = float(row['Cost'])
        toPay = []
        for p in peopleList:
            if row[p] == 1 and p != payer:
                toPay.append(p)
        cost = round(cost / (len(toPay) + 1), 2)
        for p in toPay:
            existingPayment, reversePayment = getPayments(outputDf, p, payer)
            if len(existingPayment) == 0 and len(reversePayment) == 0:
                newRow = {
                    'Need to Pay': p,
                    'Cost': cost,
                    'Pay to': payer
                }
                outputDf = outputDf._append(newRow, ignore_index = True)
            if len(existingPayment) == 1:
                mask = (outputDf['Need to Pay'] == p) & (outputDf['Pay to'] == payer)
                outputDf.loc[mask, 'Cost'] += cost
            if len(reversePayment) == 1:
                reverseCost = reversePayment['Cost'].values[0]
                mask = (outputDf['Need to Pay'] == payer) & (outputDf['Pay to'] == p)
                if reverseCost > cost:
                    outputDf.loc[mask, 'Cost'] -= cost
                else:
                    outputDf.loc[mask, 'Cost'] = cost - reverseCost
                    outputDf.loc[mask, 'Need to Pay'] = p
                    outputDf.loc[mask, 'Pay to'] = payer
    outputDf = outputDf[outputDf['Cost'] != 0]
    outputDf.sort_values(by = 'Need to Pay', inplace = True)
    return outputDf
        

initialDf = pd.read_csv(initialPath)
# transformDf = transformFormResponse(initialDf)
# transformDf.to_csv(transformPath)
# outputDf = computeCosts(transformDf, getPeopleList(initialDf))
# outputDf.to_csv(outputPath)

transformDf = pd.read_csv("C:/Users/ucg8nb/Downloads/Transformed Cost Splitty - Transformed Cost Splitty.csv")
outputDf = computeCosts(transformDf, getPeopleList(initialDf))
outputDf.to_csv(outputPath)