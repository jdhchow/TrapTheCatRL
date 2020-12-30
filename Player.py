import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers, models

'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the RL player

From Grid, player can access:
grid.createWall(index)
grid.getValidPlayerMoves()
'''


class Player:
    def __init__(self):
        self.epsilon = 0.2
        self.epsilonStepSize = 0.001
        self.gamma = 0.8

        # Set for each task
        self.gridDim = None
        self.valueFunc = None
        self.actions = None

        # Set for each game
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

    def newGame(self):
        self.prevActions = []
        self.prevStates = []

    def updateValueFunc(self, terminalReward):
        steps = len(self.prevStates)
        discountedRewards = np.array([(self.gamma ** (steps - x)) * terminalReward for x in range(steps)])
        actions = []

        for state, action in zip(self.prevStates, self.prevActions):
            newGrid = state.createWall(action)
            actions += [np.array(newGrid) / 2]   # Standardize grid for neural network

        actions = np.array(actions)
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

    def explore(self, actions):
        return random.choice(actions)

    def optimize(self, grid, actions):
        actionValues = []

        for action in actions:
            newGrid = grid.createWall(action)
            newGrid = np.array(newGrid) / 2  # Standardize grid for neural network
            newGrid = newGrid.reshape(newGrid.shape + (1,))

            actionValues += [(action, self.valueFunc.predict(np.array([newGrid])))]

        maxValue = max([actionValue[1] for actionValue in actionValues])
        optActions = [actionValue[0] for actionValue in actionValues if actionValue[1] == maxValue]

        return self.explore(optActions)

    def move(self, grid):
        actions = grid.getValidPlayerMoves()

        # Update epsilon
        self.updateEpsilon()
        action = self.explore(actions) if random.random() < self.epsilon else self.optimize(grid, actions)

        self.prevStates += [grid]
        self.prevActions += [action]

        return action
