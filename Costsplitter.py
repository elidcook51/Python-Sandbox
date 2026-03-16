import pandas as pd

#Instructions for how to create thingy

def findMatchingCell(row, column, output, people):
    newCol = people[row]
    index = 0
    for p in people:
        if(p==column):
            break
        else:
            index += 1
    newRow = index
    p1 = output.loc[row, column]
    print(newRow, newCol)
    p2 = output.loc[newRow, newCol]
    return p1, p2, newRow, newCol

def getPeopleList(df):
    return list(set((df['People?'].dropna().str.split(',').explode().str.strip().unique()).tolist()))

def transformFormResponse(df):
    peopleList = getPeopleList(df)
    outputDf = pd.DataFrame()
    for index, row in df.iterrows():
        startingDict = {name: 0 for name in peopleList}
        users = []
        users = str(row['People?']).split(',')
        for u in users:
            if u in startingDict:
                startingDict[u] = 1
        startingDict['People who paid'] = row['Who paid']
        startingDict['Cost'] = row['Cost']
        startingDict['Reason'] = row['Item?']
        outputDf = outputDf._append(startingDict, ignore_index = True)
    return outputDf



def splitCosts(data, people):
    data['Cost'] = data['Cost'].astype('float')
    data_ = []
    for person in people:
        if person != 'Name':
            temp = [person] + [0] * (len(people))
            print(temp)
            data_.append(temp)
    output = pd.DataFrame(data_, columns = [''] + people)
    output.to_csv("C:/Users/ucg8nb/Downloads/Output CSV for Eli.csv")
    for row in range(len(data)):
        owed = data.loc[row, 'People who paid']
        owe = []
        for column in data:
            if (column != 'People who paid') & (column != 'Cost') & (data.loc[row, column] == 1) & (column != owed):
                owe.append(column)
        cost = data.loc[row, 'Cost']
        for person in owe:
            row_pos = people.index(person.strip())
            col_pos = output.columns.get_loc(str(owed).strip())
            print(output.columns[col_pos])
            output.iat[row_pos, col_pos] += float(cost) / (1 + len(owe))
    for payer in range(len(output)):
        for paid in output:
            if paid != 'Name':
                p1, p2, newRow, newCol = findMatchingCell(payer, paid, output,people)
                if (p1 == 0 or p2 == 0):
                    pass
                elif p1 > p2:
                    finalOutput = p1 - p2
                    output.loc[payer, paid] = finalOutput
                    output.loc[newRow, newCol] = 0
                elif p2 > p1:
                    finalOutput = p2 - p1
                    output.loc[newRow, newCol] = finalOutput
                    output.loc[payer, paid] = 0
                elif p1 == p2:
                    output.loc[newRow, newCol] = 0
                    output.loc[payer, paid] = 0
    return output

def completePipeline(formCsvPath, outputPath):
    formCSV = pd.read_csv(formCsvPath)
    transformedDf = transformFormResponse(formCSV)
    peopleList = getPeopleList(formCSV)
    finalDf = splitCosts(transformedDf, peopleList)
    finalDf.to_csv(outputPath)

csvPath = "C:/Users/ucg8nb/Downloads/Cost Splitty Spring Break 2026 (Responses) - Form Responses 1.csv"
outputPath = 'C:/Users/ucg8nb/Downloads/First Test Cost Splitty.csv'
completePipeline(csvPath, outputPath)