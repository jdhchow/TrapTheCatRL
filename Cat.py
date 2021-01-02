from Agent import Agent, AgentKind, RLAgent
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
        q.extend([(next_, dist+1) for next_ in grid.getValidCatMoves(curr)])

    # We can't reach the edge anymore :(
    return 10000


class Cat(Agent):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)

    def validActions(self, grid):
        return grid.getValidCatMoves()

    def applyAction(self, grid, action):
        return grid.updateCatLoc(action)[0]

    @property
    def kind(self):
        return AgentKind.CAT

class RandomCat(Cat):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)

    def save(self):
        pass

    def updateValueFunc(self, terminalReward, gameNum):
        pass

    def move(self, grid):
        validMoves = grid.getValidCatMoves()   # Gives the coordinates of valid adjacent squares

        move = random.choice(validMoves)
        return move

class ShortestPathCat(Cat):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)

    def save(self):
        pass

    def updateValueFunc(self, terminalReward, gameNum):
        pass

    def move(self, grid):
        return sorted([(get_dist(grid, x), x) for x in grid.getValidCatMoves()])[0][1]

class RLCat(RLAgent, Cat):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)
