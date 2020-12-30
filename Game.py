import random


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the game
'''


class Grid:
    def __init__(self, yLen, xLen, grid=None, catLoc=None):
        self.yLen = yLen
        self.xLen = xLen
        self.totalTiles = yLen * xLen

        # Set these to import pre-existing grid
        self.grid = grid
        self.catLoc = catLoc

    def index2tuple(self, index):
        return index // self.yLen, index - self.yLen * (index // self.yLen)

    def createWall(self, index):
        col, row = self.index2tuple(index)
        self.grid[col][row] = 1

    def checkCollision(self, index):
        col, row = self.index2tuple(index)

        # Check collision with another wall or cat
        if self.grid[col][row] != 0:
            return True

        return False

    def getValidPlayerMoves(self):
        validActions = [move for move in range(self.totalTiles) if not self.checkCollision(move)]
        return validActions

    def getPotentialCatMoves(self):
        xOffsets = [-1 + self.catLoc[0] % 2, 0 + self.catLoc[0] % 2]
        offsets = [(0, -1), (0, 1)] + [(y, x) for y in [-1, 1] for x in xOffsets]
        return [tuple(sum(joined) for joined in zip(self.catLoc, offset)) for offset in offsets]

    def getValidCatMoves(self):
        potentialMoves = self.getPotentialCatMoves()
        return [moveTuple for moveTuple in potentialMoves if self.grid[moveTuple[0]][moveTuple[1]] != 1]

    def updateCatLoc(self, move):
        self.grid[self.catLoc[0]][self.catLoc[1]] = 0
        self.grid[move[0]][move[1]] = 2
        self.catLoc = move

    def newGrid(self, pFill):
        # Generate initial grid
        self.grid = [[0 for row in range(self.xLen)] for col in range(self.yLen)]

        # Flip random, unoccupied tiles
        nFill = int(self.totalTiles * pFill)
        fillIndices = random.sample(population=range(self.totalTiles), k=nFill)

        for index in fillIndices:
            self.createWall(index)

        # Place cat
        self.catLoc = (self.yLen // 2, self.xLen // 2)
        self.grid[self.catLoc[0]][self.catLoc[1]] = 2

    def displayGrid(self):
        # Print rows with offset
        for col in range(self.yLen):
            offset = '' if col % 2 == 0 else ' '
            print(offset + ' '.join([str(tile) for tile in self.grid[col]]))


class Game:
    def __init__(self, gridDim):
        self.grid = Grid(*gridDim)
        self.turn = 0
        self.winner = -1

    def newGame(self, pFill):
        # (Re)set the winner and turn
        self.winner = -1  # -1: Game ongoing, 0: Player win, 1: Cat win
        self.turn = 0

        # Generate initial grid
        self.grid.newGrid(pFill)

    def checkPlayerWin(self):
        # Check player win (cat has no moves left)
        validMoves = self.grid.getValidCatMoves()

        if not validMoves:
            self.winner = 0
            return True

        return False

    def checkCatWin(self):
        potentialMoves = self.grid.getPotentialCatMoves()
        win = False

        # Check if cat can move off the grid (y-axis)
        if not {-1, self.grid.yLen}.isdisjoint([moveTuples[0] for moveTuples in potentialMoves]):
            self.winner = 1
            win = True
        # Check if cat can move off the grid (x-axis)
        elif not {-1, self.grid.xLen}.isdisjoint([moveTuples[1] for moveTuples in potentialMoves]):
            self.winner = 1
            win = True

        return win

    def movePlayer(self, index):
        # Ensure move is valid (tile index is empty)
        assert not self.grid.checkCollision(index)

        # Increment turn
        self.turn += 1

        # Process player move
        self.grid.createWall(index)

    def moveCat(self, move):
        # Get valid cat moves (potential moves less blocked tiles)
        validMoves = self.grid.getValidCatMoves()

        # Ensure selected move is valid
        assert move in validMoves

        # Update grid and cat location
        self.grid.updateCatLoc(move)
