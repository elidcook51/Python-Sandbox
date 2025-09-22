import pandas as pd

def scoreValue(score, team, teamsCount):
   if teamsCount[team] < 7:
      return score

longCSV = pd.read_csv("C:/Users/ucg8nb/Downloads/Cavalier Invitational XC 2025 Results Raw.csv", header = None, encoding = 'latin1')
list = longCSV[0].tolist()
outputCSV = pd.DataFrame()
gender = 'M'
menTeams = ['University of Virginia', 'Virginia Tech', 'University of Maryland', 'College of William and Mary', 'Virginia Commonwealth University', 'James Madison University', 'Liberty University', 'Georgetown University']
womenTeams = ['Virginia Tech', 'University of Virginia', 'University of Maryland', 'James Madison University', 'Liberty University', 'Georgetown University']
scoreCountMen = {}
for t in menTeams:
   scoreCountMen[t] = 0
scoreCountWomen = {}
for t in womenTeams:
   scoreCountWomen[t] = 0
curTeams = menTeams
curScoreCount = scoreCountMen
score = 1
for i in range(int(len(list) / 5)):
   newDict = {}
   indexStart = i * 5
   name = list[indexStart + 1]
   nameSpace = name.index(" ")
   firstName = name[:nameSpace]
   lastName = name[nameSpace + 1:]
   team = list[indexStart + 2]
   place = list[indexStart]
   if name == "Anna Rigby":
      gender = 'F'
      curTeams = womenTeams
      curScoreCount = scoreCountWomen
      score = 0
   if team in curTeams:
      if curScoreCount[team] < 7:
         curScoreCount[team] += 1
         newDict['Score'] = score
         score += 1
   newDict['Place'] = place
   newDict['Name'] = name
   newDict['Team'] = team
   newDict['Result'] = list[indexStart + 3] + '0'
   newDict['First Name']= firstName
   newDict['Last Name'] = lastName
   newDict['Gender'] = gender
   bibThing = list[indexStart + 4] + ' '
   space = bibThing.index(' ')
   bib = bibThing[:space]
   bib = bib.replace('#', "")
   newDict['Bib'] = bib
   outputCSV = outputCSV._append(newDict, ignore_index = True)
print(outputCSV['Result'])
outputCSV.to_csv("C:/Users/ucg8nb/Downloads/Cavalier Invitational XC 2025 Results.csv")

