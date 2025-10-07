import pandas as pd

def findMatchingCell(row, column, output):
    people = ['Eli Cook', 'Aaron Falik', 'Aidan Szilagyi', 'Beall Roberts', 'Chase Cartwright', 'Dana Johannsen', 'Declan McQuinn', 'Ella Harris', 'Hayley Sandler', 'Jackie Janicki', 'Jenny Macler', 'Jesse Smith', 'Kamren Reeves', 'Lisa de Groot', 'Mitchell Palmer', 'Tyler Jackson', 'Will Taylor']
    newCol = people[row]
    index = 0
    for p in people:
        if(p==column):
            break
        else:
            index += 1
    newRow = index
    p1 = output.loc[row, column]
    p2 = output.loc[newRow, newCol]
    return p1, p2, newRow, newCol

fileName = "C:/Users/ucg8nb/Downloads/Cost Splitty Boone - Sheet1.csv"
data = pd.read_csv(fileName)
people = ['Name', 'Eli Cook', 'Aaron Falik', 'Aidan Szilagyi', 'Beall Roberts', 'Chase Cartwright', 'Dana Johannsen',
          'Declan McQuinn', 'Ella Harris', 'Hayley Sandler', 'Jackie Janicki', 'Jenny Macler', 'Jesse Smith',
          'Kamren Reeves', 'Lisa de Groot', 'Mitchell Palmer', 'Tyler Jackson', 'Will Taylor']
data['Cost'] = data['Cost'].astype('float')
data_ = []
for person in people:
    if person != 'Name':
        temp = [person, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0]
        data_.append(temp)
output = pd.DataFrame(data_, columns = people)
for row in range(len(data)):
    owed = data.loc[row, 'People who paid']
    owe = []
    for column in data:
        if (column != 'People who paid') & (column != 'Cost') & (data.loc[row, column] == 1) & (column != owed):
            owe.append(column)
    cost = data.loc[row, 'Cost']
    for person in owe:
        index = -1
        for i in people:
            if person == i:
                break
            else:
                index += 1
        output.loc[index, owed] += (cost + 0.0) / (1 + len(owe))
for payer in range(len(output)):
    for paid in output:
        if paid != 'Name':
            p1, p2, newRow, newCol = findMatchingCell(payer, paid, output)
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
output.to_csv('C:/Users/ucg8nb/Downloads/finalResult Boone.csv', encoding = 'utf-8', index = False)
