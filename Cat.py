import random

from Game import Grid  # Cat can access grid.updateCatLoc(move) and grid.getValidCatMoves()


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the cat
'''


class Cat:
    def __init__(self):
        self.gridDim = None

    def newTask(self, gridDim, valueFuncPath=None):
        self.gridDim = gridDim

    def newGame(self):
        pass

    def updateValueFunc(self, terminalReward):
        pass

    def move(self, state, catLoc):
        grid = Grid(*self.gridDim, state, catLoc)
        validMoves = grid.getValidCatMoves()   # Gives the coordinates of valid adjacent squares

        move = random.choice(validMoves)
        return move
