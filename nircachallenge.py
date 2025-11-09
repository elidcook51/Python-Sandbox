import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import time, timedelta

# url = 'https://clubrunning.org/races/race_roster.php?race=1296'

# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')

# tables = soup.find_all('table', id = 'matrix')

# race = ['Championship', 'Freshman/Sophmore', 'Junior/Senior/Grad']
# gender = ['Male', 'Female']

# outputDf = pd.DataFrame()

# for table in tables:
#     rows = table.find_all('tr')[1:]
#     for row in rows:
#         cols = row.find_all('td')
#         if len(cols) == 0:
#             continue
#         newRow = {
#             'Club': cols[2].get_text(strip = True),
#             'Last Name': cols[3].get_text(strip = True),
#             'First Name': cols[4].get_text(strip = True),
#             'Athlete Class': cols[5].get_text(strip = True),
#             'Bib': cols[6].get_text(strip = True)
#         }
#         outputDf = outputDf._append(newRow, ignore_index = True)
# outputDf.to_csv("C:/Users/ucg8nb/Downloads/Races Signups.csv")

# collegeSignups = pd.read_csv("C:/Users/ucg8nb/Downloads/Races Signups.csv")

# open6kSignup = pd.read_csv("C:/Users/ucg8nb/Downloads/20251107-NIRCACrossCountryOpen6K-participants.csv")


# open6kSignup['First Name'] =  open6kSignup['First Name'].str.lower()
# open6kSignup['Last Name'] = open6kSignup['Last Name'].str.lower()

# collegeSignups['First Name'] = collegeSignups['First Name'].str.lower()
# collegeSignups['Last Name'] = collegeSignups['Last Name'].str.lower()

# commonName = pd.merge(collegeSignups, open6kSignup, on = ['Last Name', 'First Name'])
# commonName.to_csv("C:/Users/ucg8nb/Downloads/challengeSignups.csv")

# link = ''

# response = requests.get(link)
# soup = BeautifulSoup(response.text, parser = 'html.parser')

# tables = soup.find_all('table', id = '')


commonName = pd.read_csv("C:/Users/ucg8nb/Downloads/challengeSignups.csv")

lastNames = commonName['Last Name'].tolist()
firstNames = commonName['First Name'].tolist()

open6kresultsLink = "C:/Users/ucg8nb/Downloads/2024 results export.xlsx"

results6k = pd.read_excel(open6kresultsLink)

challenge6kResults = results6k[results6k['Last Name'].isin(lastNames)]
challenge6kResults = challenge6kResults[challenge6kResults['First Name'].isin(firstNames)]

collegeResultLink = "C:/Users/ucg8nb/Downloads/2025 results excel.xlsx"

resultsCollege = pd.read_excel(collegeResultLink)
resultsCollege['First Name'] = resultsCollege['First Name'].str.lower()
resultsCollege['Last Name'] = resultsCollege['Last Name'].str.lower()

challengeResultsCollege = resultsCollege[resultsCollege['Last Name'].isin(lastNames)]
challengeResultsCollege = challengeResultsCollege[challengeResultsCollege['First Name'].isin(firstNames)]

# firstNames = ['Charles', 'Eli']
# lastNames = ['Melvin', 'Cook']

outputDf = pd.DataFrame()

for (firstName, lastName) in zip(firstNames, lastNames):
    tempDf = resultsCollege[resultsCollege['First Name'] == firstName]
    tempDf = tempDf[tempDf['Last Name'] == lastName]
    if len(tempDf) < 2:
        continue
    openTime = tempDf.loc[tempDf['Race '] == 1, 'Finish Time'].values[0]
    collegeTime = tempDf.loc[tempDf['Race '] != 1, 'Finish Time'].values[0]
    delta1 = timedelta(hours = openTime.hour, minutes = openTime.minute, seconds = openTime.second)
    delta2 = timedelta(hours = collegeTime.hour, minutes = collegeTime.minute, seconds = collegeTime.second)
    # print(delta1, delta2, delta1 + delta2)
    newRow = {
        'First Name': firstName,
        'Last Name': lastName,
        'Gender': tempDf['Gender'].values[0],
        'Team': tempDf['School'].values[0],
        'College Race time': collegeTime,
        'Open 6k Time': openTime,
        'Total Time': delta1 + delta2
    }
    outputDf = outputDf._append(newRow, ignore_index = True)

mensResults = outputDf[outputDf['Gender'] == 'M']
womensResults = outputDf[outputDf['Gender'] == 'F']

mensResults = mensResults.sort_values(by = 'Total Time', ascending = True)
womensResults = womensResults.sort_values(by = 'Total Time', ascending = True)

mensResults.to_csv("C:/Users/ucg8nb/Downloads/Mens Results.csv")
womensResults.to_csv("C:/Users/ucg8nb/Downloads/Womens Results.csv")