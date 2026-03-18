from itertools import combinations

class GameBoard():

    def __init__(self, shapeList: list[int], holePos: tuple[int, int]):
        self.grid = []
        for dim in shapeList:
            self.grid.append([1] * dim)
        x, y = holePos
        self.grid[y][x] = 0

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
        return len(self.getListOfOnes)
    
    def playAllGames(self):
        bestGame = []

    def __str__(self):
        outputList = ""
        for y in self.grid:
            for x in y:
                outputList += f"{x} "
            outputList += '\n'
        return outputList

newGame = GameBoard([1,2,3,4], (0,0))
print(newGame.legalMoveOptions())
# newGame.performMove((2,2), (1,1))