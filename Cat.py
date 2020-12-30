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
        self.valueFunc = None

    def newTask(self, gridDim, valueFuncPath=None):
        self.gridDim = gridDim

    def newGame(self):
        raise NotImplementedError

    def updateValueFunc(self, terminalReward):
        raise NotImplementedError

    def move(self, state, catLoc):
        raise NotImplementedError

class RandomCat(Cat):
    def __init__(self):
        super().__init__()

    def newGame(self):
        pass

    def updateValueFunc(self, terminalReward):
        pass

    def move(self, state, catLoc):
        grid = Grid(*self.gridDim, state, catLoc)
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

    def move(self, state, catLoc):
        return sorted([(get_dist(x), x) for x in state.getValidCatMoves()])[0][1]
