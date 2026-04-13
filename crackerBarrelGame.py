from itertools import combinations
from copy import deepcopy

#Requires game board to have a left edge that all points must align with
class GridGameBoard():

    def __init__(self, shapeList: list[int], holePos: tuple[int, int]):
        self.grid = []
        for dim in shapeList:
            self.grid.append([1] * dim)
        x, y = holePos
        self.grid[y][x] = 0
        self.startingGrid = deepcopy(self.grid)

    def computeMoveSpot(self, stick1, stick2):
        if stick1[0] == stick2[0]:
            if stick1[1] > stick2[1]:
                return (stick1[0], stick2[1] - 1)
            else:
                return (stick1[0], stick2[1] + 1)
        if stick1[1] == stick2[1]:
            if stick1[0] > stick2[0]:
                return (stick2[0] - 1, stick2[1])
            else:
                return (stick2[0] + 1, stick2[1])
        if stick1[0] > stick2[0]:
            if stick1[1] > stick2[1]:
                return (stick2[0] - 1, stick2[1] - 1)
            else:
                return (stick2[0] - 1, stick2[1] + 1)
        else:
            if stick1[1] > stick2[1]:
                return (stick2[0] + 1, stick2[1] - 1)
            else:
                return (stick2[0] + 1, stick2[1] + 1)

    def isLegalMove(self, stick1, stick2):
        if stick1 == stick2:
            return False
        stick1x, stick1y = stick1
        stick2x, stick2y = stick2
        if self.grid[stick1y][stick1x] == 0 or self.grid[stick2y][stick2x] == 0:
            return False
        if abs(stick1x - stick2x) > 1 or abs(stick1y - stick2y) > 1:
            return False
        newX, newY = self.computeMoveSpot(stick1, stick2)
        if newY < 0 or newY > len(self.grid) - 1:
            return False
        if newX < 0 or newX > len(self.grid[newY]) - 1:
            return False
        if self.grid[newY][newX] != 0:
            return False
        return True
        
    def performMove(self, move):
        stick1, stick2 = move
        if not self.isLegalMove(stick1, stick2):
            print("Illegal Move Attempt")
            return None
        newX, newY = self.computeMoveSpot(stick1, stick2)
        self.grid[newY][newX] = 1
        self.grid[stick1[1]][stick1[0]] = 0
        self.grid[stick2[1]][stick2[0]] = 0
        return None

    def getListOfOnes(self):
        outputList = []

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == 1:
                    outputList.append((j,i))
        return outputList

    def legalMoveOptions(self) -> list[tuple[int, int]]:
        possiblePlaces = self.getListOfOnes()
        outputList = []
        for comb in list(combinations(possiblePlaces, 2)):
            if self.isLegalMove(comb[0], comb[1]):
                outputList.append(comb)
            if self.isLegalMove(comb[1], comb[0]):
                outputList.append((comb[1], comb[0]))
        return outputList

    def getFinalScore(self):
        return len(self.getListOfOnes())
    
    def saveGameState(self):
        return deepcopy(self.grid)
    
    def loadGameState(self, grid):
        self.grid = deepcopy(grid)
    
    def playAllMoves(self, previousMoves = [], bestScore = float("inf"), bestMoves = []):
        if len(self.legalMoveOptions()) == 0:
            if self.getFinalScore() < bestScore:
                return previousMoves, self.getFinalScore()
            else:
                return bestMoves, bestScore
        for move in self.legalMoveOptions():
            tempGrid = self.saveGameState()
            self.performMove(move)
            tempPreviousMoves = deepcopy(previousMoves)
            tempPreviousMoves.append(move)
            bestMoves, bestScore = self.playAllMoves(tempPreviousMoves, bestScore, bestMoves)
            self.loadGameState(tempGrid)
        return bestMoves, bestScore

    def playSequenceOfMoves(self, moveSequence):
        for move in moveSequence:
            self.performMove(move)
            print(str(self))

    def resetBoard(self):
        self.loadGameState(self.startingGrid)

    def __str__(self):
        outputList = ""
        for y in self.grid:
            for x in y:
                outputList += f"{x} "
            outputList += '\n'
        return outputList

class GeneralGridGameBoard():

    #Details on what board should look like
    #The board is a 2d array of 0, 1, and -1
    #0 or 1 means pin or not pin respectively, -1 means invalid spot
    #Example:
    """
    board = [
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  0,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
    ]
    """

    def __init__(self, grid : list[list[int]], hole = None):
        self.grid = grid
        if hole:
            self.grid[hole[0]][hole[1]] = 0
        self.startingGrid = deepcopy(self.grid)
    
    def computeMoveSpot(self, stick1, stick2):
        if stick1[0] == stick2[0]:
            if stick1[1] > stick2[1]:
                return (stick1[0], stick2[1] - 1)
            else:
                return (stick1[0], stick2[1] + 1)
        if stick1[1] == stick2[1]:
            if stick1[0] > stick2[0]:
                return (stick2[0] - 1, stick2[1])
            else:
                return (stick2[0] + 1, stick2[1])
        if stick1[0] > stick2[0]:
            if stick1[1] > stick2[1]:
                return (stick2[0] - 1, stick2[1] - 1)
            else:
                return (stick2[0] - 1, stick2[1] + 1)
        else:
            if stick1[1] > stick2[1]:
                return (stick2[0] + 1, stick2[1] - 1)
            else:
                return (stick2[0] + 1, stick2[1] + 1)
        
    def isLegalMove(self, stick1, stick2):
        if stick1 == stick2:
            return False
        stick1x, stick1y = stick1
        stick2x, stick2y = stick2
        if self.grid[stick1y][stick1x] == 0 or self.grid[stick2y][stick2x] == 0:
            return False
        if abs(stick1x - stick2x) > 1 or abs(stick1y - stick2y) > 1:
            return False
        newX, newY = self.computeMoveSpot(stick1, stick2)
        if newX > len(self.grid[0]) - 1 or newX < 0:
            return False
        if newY > len(self.grid) - 1 or newY < 0:
            return False
        if self.grid[newY][newX] == 0:
            return True
        return False
    
    def performMove(self, move):
        stick1, stick2 = move
        if not self.isLegalMove(stick1, stick2):
            print("Illegal Move Attempt")
            return None
        newX, newY = self.computeMoveSpot(stick1, stick2)
        self.grid[newY][newX] = 1
        self.grid[stick1[1]][stick1[0]] = 0
        self.grid[stick2[1]][stick2[0]] = 0
        return None

    def getListOfSticks(self):
        outputList = []

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == 1:
                    outputList.append((j,i))
        return outputList

    def legalMoveOptions(self) -> list[tuple[int, int]]:
        possiblePlaces = self.getListOfSticks()
        outputList = []
        for comb in list(combinations(possiblePlaces, 2)):
            if self.isLegalMove(comb[0], comb[1]):
                outputList.append(comb)
            if self.isLegalMove(comb[1], comb[0]):
                outputList.append((comb[1], comb[0]))
        return outputList
    
    def getFinalScore(self):
        return len(self.getListOfSticks())
    
    def saveGameState(self):
        return deepcopy(self.grid)
    
    def loadGameState(self, grid):
        self.grid = deepcopy(grid)

    def playAllMoves(self, previousMoves = [], bestScore = float("inf"), bestMoves = []):
        if len(self.legalMoveOptions()) == 0:
            if self.getFinalScore() < bestScore:
                return previousMoves, self.getFinalScore()
            else:
                return bestMoves, bestScore
        for move in self.legalMoveOptions():
            tempGrid = self.saveGameState()
            self.performMove(move)
            tempPreviousMoves = deepcopy(previousMoves)
            tempPreviousMoves.append(move)
            bestMoves, bestScore = self.playAllMoves(tempPreviousMoves, bestScore, bestMoves)
            self.loadGameState(tempGrid)
        return bestMoves, bestScore

    def playSequenceOfMoves(self, moveSequence):
        for move in moveSequence:
            self.performMove(move)
            print(str(self))

    def resetBoard(self):
        self.loadGameState(self.startingGrid)

    def __str__(self):
        outputList = ""
        for y in self.grid:
            for x in y:
                if x == -1:
                    outputList += f"{x} "
                else:
                    outputList += f" {x} "
            outputList += '\n'
        return outputList

testBoard =  [
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  0,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
    ]

testBoard = GeneralGridGameBoard(testBoard)
solution, bestScore = testBoard.playAllMoves()
print(solution)