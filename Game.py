import random


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the game
'''


class Game:
    def __init__(self):
        self.grid = None
        self.catLoc = None
        self.gridDim = None
        self.winner = None

    def createWall(self, index):
        col = index // self.gridDim[0]
        row = index - self.gridDim[0] * (index // self.gridDim[0])

        # Check collision with cat
        if self.grid[col][row] != 2:
            self.grid[col][row] = 1

    def newGame(self, y, x, pFill):
        # (Re)set the winner
        self.winner = 1

        # Generate initial grid
        self.grid = [[0 for row in range(x)] for col in range(y)]
        self.gridDim = (y, x)

        # Flip random, unoccupied tiles
        totalTiles = y * x
        nFill = int(totalTiles * pFill)
        fillIndices = random.sample(population=range(totalTiles), k=nFill)

        for index in fillIndices:
            self.createWall(index)

        # Place cat
        self.catLoc = (y // 2, x // 2)
        self.grid[self.catLoc[0]][self.catLoc[1]] = 2

    def displayGrid(self):
        for col in range(self.gridDim[0]):
            offset = '' if col % 2 == 0 else ' '
            print(offset + ' '.join([str(tile) for tile in self.grid[col]]))

    def checkCatWin(self, potentialMoves):
        # Check if cat is on edge of grid
        for i in range(2):
            if not {-1, self.gridDim[i]}.isdisjoint([moveTuples[i] for moveTuples in potentialMoves]):
                return True

        return False

    def move(self):
        # List surrounding tiles
        hardOffset = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        potentialMoves = [tuple(sum(joined) for joined in zip(self.catLoc, offset)) for offset in hardOffset]

        # Check win by cat
        if self.checkCatWin(potentialMoves):
            self.winner = -1
            return

        # Remove blocked moves
        potentialMoves = [moveTuple for moveTuple in potentialMoves if self.grid[moveTuple[0]][moveTuple[1]] != 1]

        # Check player win
        if not potentialMoves:
            self.winner = 0
            return

        # Select move
        move = potentialMoves[0]

        # Update grid and cat location
        self.grid[self.catLoc[0]][self.catLoc[1]] = 0
        self.grid[move[0]][move[1]] = 2
        self.catLoc = move
