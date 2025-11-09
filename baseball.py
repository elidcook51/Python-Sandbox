import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
import ast
import copy
import os

def buildX(X, featureTuple):
    selectedFeatues, useConstant = featureTuple
    if useConstant:
        return sm.add_constant(X[selectedFeatues])
    else:
        return X[selectedFeatues]

def forwardStepRegression(X, y):
    selectedFeatures = []
    useConstant = False    
    remainingFeatures = list(X.columns)
    curScore = float('inf')

    while len(remainingFeatures) > 0:
        candidateScores = []
        for candidate in remainingFeatures:
            testFeatures = selectedFeatures + [candidate]
            testX = buildX(X, (testFeatures, useConstant))
            model = sm.OLS(y, testX).fit()
            score = model.aic
            candidateScores.append((score, candidate))
        
        if not useConstant:
            constantX = sm.add_constant(X[selectedFeatures])
            score = sm.OLS(y, constantX).fit().aic
            candidateScores.append((score, 'Constant'))

        candidateScores.sort(key = lambda x: x[0])
        newScore, newCandidate = candidateScores[0]

        if newScore < curScore:
            if newCandidate == 'Constant':
                useConstant = True
            else:
                selectedFeatures.append(newCandidate)
                remainingFeatures.remove(newCandidate)
            curScore = newScore
        else:
            break
    modelX = buildX(X, (selectedFeatures, useConstant))

    return sm.OLS(y, modelX)

class Team:

    def __init__(self, salaryCap = 30000000):
        self.salaryCap = salaryCap
        self.curSalary = 0
        self.players = []
        self.opsList = []
    
    def pickPlayer(self, playerNumber, salary, ops):
        if self.curSalary + salary > self.salaryCap:
            return 'Improper Draft'
        else:
            self.players.append(playerNumber)
            self.curSalary += salary
            self.opsList.append(ops)
            return "Worked"

    def getPlayers(self):
        return self.players

    def getOPSAvg(self):
        return np.mean(self.opsList)
    
    def __str__(self):
        return f"Player #s selected: {self.players}, with salary of {self.curSalary} used"

class TeamDrafts:

    def __init__(self, numTeams, playerData, teamData):
        self.teamList = []
        self.numTeams = numTeams
        for i in range(numTeams):
            newTeam = Team()
            self.teamList.append(newTeam)
        self.curPlace = 0
        self.playerData = playerData
        self.updatingPlayerData = copy.deepcopy(self.playerData)
        self.forward = True
        self.teamData = teamData
        self.playerList = self.playerData['Player #'].tolist()
        self.playerNames = self.playerData['Player'].tolist()
        self.playerSalaries = self.playerData['Salary'].tolist()
        self.playerOPS = self.playerData['OPS'].tolist()
        self.draftedPlayersList = []
        
    def getPlayerSalary(self, playerNumber):
        return self.playerData.loc[self.playerData['Player #'] == playerNumber]['Salary'].values[0]
    
    def getPlayerOPS(self, playerNumber):
        return self.playerData.loc[self.playerData['Player #'] == playerNumber]['OPS'].values[0]

    def draft(self, playerNumber):
        if playerNumber == -1:
            if self.forward:
                self.curPlace += 1
                if self.curPlace == len(self.teamList):
                    self.forward = False
                    self.curPlace -= 1
            else:
                self.curPlace -= 1
                if self.curPlace < 0:
                    self.forward = True
                    self.curPlace = 0
            return None
        worked = self.teamList[self.curPlace].pickPlayer(int(playerNumber), self.getPlayerSalary(playerNumber), self.getPlayerOPS(playerNumber))
        if worked != 'Improper Draft':
            if self.forward:
                self.curPlace += 1
                if self.curPlace == len(self.teamList):
                    self.forward = False
                    self.curPlace -= 1
            else:
                self.curPlace -= 1
                if self.curPlace < 0:
                    self.forward = True
                    self.curPlace = 0
        self.draftedPlayersList.append(playerNumber)
        
    def getCoefs(self):
        X = self.teamData.drop(columns = ['team', 'team_num', 'Win'])

        y = self.teamData['Win']

        forwardModel = forwardStepRegression(X, y).fit()

        rbi = self.teamData['RBI']
        rbiX = teamData.drop(columns = ['team', 'team_num', 'ERA', 'Win', 'RBI', 'TB', 'SLG', 'OBP'])

        rbiModel = forwardStepRegression(rbiX, rbi).fit()

        self.rbiCoef = forwardModel.params['RBI']
        self.opsCoef = rbiModel.params['OPS']
    
    def createWholePlayerRankings(self):
        outputDf = pd.DataFrame()
        for comb in list(combinations(self.playerList, 4)):
            ops = []
            totCost = []
            names = []
            for player in comb:
                row = playerData.loc[playerData['Player #'] == player]
                ops.append(row['OPS'])
                totCost.append(row['Salary'])
                names.append(row['Player'].values[0])
            if np.sum(totCost) < 30000000 and np.min(ops) > 0.707:
                newRow = {
                    'Players': names,
                    'Player Numbers': list(comb),
                    'Average OPS': np.mean(ops),
                    'Total Cost': np.sum(totCost),
                }
                outputDf = outputDf._append(newRow, ignore_index = True)
        outputDf.to_csv("C:/Users/ucg8nb/Downloads/All Player Combinations.csv")
        
    def rankToWeight(self, rank):
        if rank < 100:
            return 1
        if rank < 500:
            return 0.4
        if rank < 1000:
            return 0.2
        if rank < 1500:
            return 0.05
        else:
            return 0.01

    def __str__(self):
        outputString = f''
        for i in range(len(self.teamList)):
            team = self.teamList[i]
            outputString += f"Team number {i}: {team} \n"
        return outputString

    def findPlayerRankings(self, onTeam, initial = False):
        teamOptions = pd.read_csv("C:/Users/ucg8nb/Downloads/All Player Combinations.csv")
        teamOptions['Player Numbers'] = teamOptions['Player Numbers'].apply(ast.literal_eval)
        for p in onTeam:
            if len(teamOptions) == 0:
                return None
            teamOptions = teamOptions[teamOptions['Player Numbers'].apply(lambda x: p in x)]
        for p in self.draftedPlayersList:
            if int(p) not in onTeam:
                if len(teamOptions) == 0:
                    return None
                teamOptions = teamOptions[teamOptions['Player Numbers'].apply(lambda x: p not in x)]
        teamOptions.sort_values(['Average OPS'], inplace = True, ignore_index = True, ascending = False)
        teamOptions.head(2000 - 30 * len(self.draftedPlayersList))

        if len(teamOptions) == 0:
            return None

        bestOPS = teamOptions['Average OPS'].values[0]

        teamOptions['Rank'] = range(1, len(teamOptions) + 1)
        teamOptions['Weight'] = (teamOptions['Average OPS'] / bestOPS) * teamOptions['Rank'].apply(self.rankToWeight)

        playerRanks = []

        drafted = []

        for playerNum in self.playerList:
            playerTeamsDf = teamOptions[teamOptions['Player Numbers'].apply(lambda x: playerNum in x)]
            if playerNum in self.draftedPlayersList:
                drafted.append("Drafted")
                playerRanks.append(-1)
            else:
                drafted.append('Not Drafted')
                playerRanks.append(np.sum(playerTeamsDf['Weight']))

        outputDf = pd.DataFrame({
            'Player': self.playerNames,
            'Player #': self.playerList,
            'Score': playerRanks,
            'Salary': self.playerSalaries,
            'OPS': self.playerOPS,
            'Status': drafted,
        })

        outputDf.sort_values('Score', ascending = False, inplace = True, ignore_index= True)
        if initial:
            outputDf.to_csv("C:/Users/ucg8nb/Downloads/Overall Player Rankings.csv")
        else:
            outputDf.to_csv("C:/Users/ucg8nb/Downloads/Current Team Player Rankings.csv")
            return outputDf
        
    
    def getBestPlayer(self):
        curTeam = self.teamList[self.curPlace]
        onTeam = curTeam.getPlayers()
        playerRankingDf = self.findPlayerRankings(onTeam)
        if playerRankingDf is None:
            return -1
        playerNumber = playerRankingDf['Player #'].values[0]
        playerName = playerRankingDf['Player'].values[0]
        playerSalary = playerRankingDf['Salary'].values[0]
        print(f"Pick {playerName}, number {playerNumber}, with salary of {playerSalary}")
        return playerNumber
    
    def setUp(self):
        filepath = "C:/Users/ucg8nb/Downloads/All Player Combinations.csv"
        if not os.path.exists(filepath):
            self.createWholePlayerRankings()
        self.findPlayerRankings([], initial = True)

    def scoreTeams(self):
        teams = []
        opsAvg = []
        for i in range(len(self.teamList)):
            teams.append(f"Team {i}")
            opsAvg.append(f"OPS Average: {self.teamList[i].getOPSAvg()}")
        opsAvg, teams = zip(*sorted(zip(opsAvg, teams)))
        for i in range(len(teams)):
            print(f"{teams[i]} with OPS Average: {opsAvg[i]}")
        return teams, opsAvg




teamData = pd.read_excel("C:/Users/ucg8nb/Downloads/baseball_data.xlsx", sheet_name = 'team.data')
playerData = pd.read_excel("C:/Users/ucg8nb/Downloads/baseball_data.xlsx", sheet_name = 'player.data')

teamDrafter = TeamDrafts(10, playerData, teamData)
# teamDrafter.setUp()

#First 10
teamDrafter.draft(14)
teamDrafter.draft(41)
teamDrafter.draft(13)
teamDrafter.draft(24)
teamDrafter.draft(31)
teamDrafter.draft(37)
teamDrafter.draft(4)
teamDrafter.draft(26)
teamDrafter.draft(18)
teamDrafter.draft(49)

#Second 10 starting at team 10
teamDrafter.draft(46)
teamDrafter.draft(42)
teamDrafter.draft(19)
teamDrafter.draft(2)
teamDrafter.draft(16)
teamDrafter.draft(1)
teamDrafter.draft(40)
teamDrafter.draft(9)
teamDrafter.draft(28)
teamDrafter.draft(25)

# print(teamDrafter)
# teamDrafter.scoreTeams()

#Third 10 starting at 1
teamDrafter.draft(17)
teamDrafter.draft(48)
teamDrafter.draft(34)
teamDrafter.draft(32)
teamDrafter.draft(22)
teamDrafter.draft(21)
teamDrafter.draft(6)
teamDrafter.draft(29)
teamDrafter.draft(7)
teamDrafter.draft(27)
# teamDrafter.scoreTeams()

#Fourth 10 starting at 10
teamDrafter.draft(38)
teamDrafter.draft(11)
teamDrafter.draft(23)
teamDrafter.draft(45)
teamDrafter.draft(43)
teamDrafter.draft(8)
teamDrafter.draft(39)
teamDrafter.draft(30)
teamDrafter.draft(15)
teamDrafter.draft(44)

print(teamDrafter)

teamDrafter.scoreTeams()
# teamDrafter.getBestPlayer()

# X = teamData.drop(columns = ['team', 'team_num', 'Win', 'RBI'])

# y = teamData['Win']

# forwardModel = forwardStepRegression(X, y).fit()

# print(forwardModel.summary())

# rbi = teamData['RBI']
# rbiX = teamData.drop(columns = ['team', 'team_num', 'ERA', 'Win', 'RBI', 'TB', 'SLG', 'OBP'])

# rbiModel = forwardStepRegression(rbiX, rbi).fit()

# print(rbiModel.summary())

# rbiCoef = forwardModel.params['RBI']
# opsCoef = rbiModel.params['OPS']











    



