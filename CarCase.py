import pandas as pd
import numpy as np

fullData = pd.read_csv("C:/Users/ucg8nb/Downloads/Car Competition Submission (Responses) - Form Responses 1.csv")

standard = ['13 wood matches in a metal screwtop waterproof container', 'A hand axe', 'A 20 x 20 piece of heavy duty canvas', 'A sleeping bag per person (arctic type down (vegan) filled with liner)', 'A gallon of maple syrup', '250ft of ¼ in braided nylon rope, 50lb test', '3 pairs of snowshoes', 'One aircraft inner tube for a 14 inch wheel (punctured)', 'Safety razor shaving kit with mirror', 'An operating 4 battery flashlight', 'A fifth of Bacardi rum (151 proof)', 'A wind-up alarm clock', 'A magnetic compass', 'A book entitled, Northern Star Navigation', 'A bottle of water purification tablets']

rawSum = []
squaredSum = []

for index, row in fullData.iterrows():
    sum = 0
    squared = 0
    for i in range(15):
        columnName = f" [{i + 1}]"
        item = row[columnName]
        properRank = standard.index(item)
        sum += abs(i - properRank)
        squared += np.power(i - properRank, 2)
    rawSum.append(sum)
    squaredSum.append(squared)

fullData['Raw Difference Sum'] = rawSum
fullData['Square Difference Sum'] = squaredSum
fullData.to_csv("C:/Users/ucg8nb/Downloads/Updated Car Rankings.csv")