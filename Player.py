from Agent import Agent, AgentKind, RLAgent

'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the RL player

From Grid, player can access:
grid.createWall(index)
grid.getValidPlayerMoves()
'''

class Player(Agent):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)
    
    def validActions(self, grid):
        return grid.getValidPlayerMoves()

    def applyAction(self, grid, action):
        return grid.createWall(action)

    @property 
    def kind(self):
        return AgentKind.PLAYER

class RLPlayer(RLAgent, Player):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)