import numpy as np
import random
import copy
from tensorflow import keras
from tensorflow.keras import layers, models


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the RL player
'''


class Player:
    def __init__(self):
        self.epsilon = 0.2
        self.epsilonStepSize = 0.001
        self.gamma = 0.8
        self.gridDim = None

        self.valueFunc = None
        self.actions = None

        self.prevStates = None
        self.prevActions = None

    def updateEpsilon(self):
        self.epsilon = 1 / (1 / self.epsilon + self.epsilonStepSize)

    def newTask(self, gridDim, valueFuncPath=None):
        self.gridDim = gridDim

        if valueFuncPath is None:
            self.valueFunc = self.buildModel(gridDim)
        else:
            self.valueFunc = keras.models.load_model(valueFuncPath)

        self.actions = list(range(np.prod(gridDim)))

        self.prevStates = []
        self.prevActions = []

    def newEpoch(self):
        self.prevActions = []
        self.prevStates = []

    def updateValueFunc(self, terminalReward):
        steps = len(self.prevStates)
        discountedRewards = np.array([(self.gamma ** (steps - x)) * terminalReward for x in range(steps)])
        actions = np.array([self.createWall(state, action) for state, action in zip(self.prevStates, self.prevActions)])
        actions = actions.reshape(actions.shape + (1,))

        self.valueFunc.train_on_batch(actions, discountedRewards)

    def buildModel(self, gridDim):
        model = models.Sequential()

        model.add(keras.Input(shape=(gridDim + (1,))))

        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        return model

    # Player knows the consequences of their actions
    def createWall(self, grid, index):
        col = index // self.gridDim[0]
        row = index - self.gridDim[0] * (index // self.gridDim[0])

        # Check collision with cat
        if grid[col][row] != 1.0:
            grid[col][row] = 0.5

        return grid

    def checkCollision(self, grid, index):
        col = index // self.gridDim[0]
        row = index - self.gridDim[0] * (index // self.gridDim[0])

        # Check collision with another wall or cat
        if grid[col][row] != 0.0:
            return True

        return False

    def explore(self, grid, actions):
        bActions = [action for action in actions if not self.checkCollision(grid, action)]
        return random.choice(bActions)

    def optimize(self, grid, actions):
        bActions = [action for action in actions if not self.checkCollision(grid, action)]
        actionVals = []

        for action in bActions:
            newGrid = self.createWall(copy.deepcopy(grid), action)
            newGrid = newGrid.reshape(newGrid.shape + (1,))
            actionVals += [(action, self.valueFunc.predict(np.array([newGrid])))]

        maxVal = max([actionVal[1] for actionVal in actionVals])
        optActions = [actionVal[0] for actionVal in actionVals if actionVal[1] == maxVal]

        return self.explore(grid, optActions)

    def move(self, grid):
        # Standardize the grid for the neural network
        grid = np.array(grid) / 2

        # Update epsilon
        self.updateEpsilon()
        index = self.explore(grid, self.actions) if random.random() < self.epsilon else self.optimize(grid, self.actions)

        self.prevStates += [grid]
        self.prevActions += [index]

        return index
