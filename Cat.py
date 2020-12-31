import random


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the cat

From Grid, cat can access:
grid.updateCatLoc(move)
grid.getPotentialCatMoves()
grid.getValidCatMoves()
grid.isWinningCatPosition(position)
'''


class Cat:
    def __init__(self):
        self.gridDim = None
        self.valueFunc = None

    def newTask(self, gridDim, valueFuncPath=None, train=True):
        self.gridDim = gridDim

    def newGame(self):
        raise NotImplementedError

    def updateValueFunc(self, terminalReward):
        raise NotImplementedError

    def move(self, grid):
        raise NotImplementedError


class RandomCat(Cat):
    def __init__(self):
        super().__init__()

    def newGame(self):
        pass

    def updateValueFunc(self, terminalReward):
        pass

    def move(self, grid):
        validMoves = grid.getValidCatMoves()   # Gives the coordinates of valid adjacent squares

        move = random.choice(validMoves)
        return move


def get_dist(grid, pos):
    q = [(pos, 0)]
    seen = set()
    while q:
        curr, dist = q.pop(0)
        if curr in seen:
            continue
        seen.add(curr)
        if grid.isWinningCatPosition(curr):
            return dist
        q.extend([(next_, dist+1) for next_ in grid.getPotentialCatMoves()])

    # We can't reach the edge anymore :(
    return 10000


class ShortestPathCat(Cat):
    def __init__(self):
        super().__init__()

    def newGame(self):
        pass

    def updateValueFunc(self, terminalReward):
        pass

    def move(self, grid):
        return sorted([(get_dist(grid, x), x) for x in grid.getValidCatMoves()])[0][1]
