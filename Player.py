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
        self.epsilon = 0.9
        self.epsilonStepSize = 0.0001
        self.gamma = 0.8

        # Set for each task
        self.gridDim = None
        self.valueFunc = None
        self.actions = None
        self.train = True

        # Set for each game
        self.prevStates = []

        # Set for some window of games
        self.prevStateRewards = []

    def updateEpsilon(self):
        self.epsilon = 1 / (1 / self.epsilon + self.epsilonStepSize)

    def newTask(self, gridDim, valueFuncPath=None, train=True):
        self.gridDim = gridDim
        self.train = train

        if valueFuncPath is None:
            self.valueFunc = self.buildModel(gridDim)
        else:
            self.valueFunc = keras.models.load_model(valueFuncPath)

    def updateValueFunc(self, terminalReward, gameNum):
        if not self.train:
            return

        steps = len(self.prevStates)
        discountedRewards = [(self.gamma ** (steps - x)) * terminalReward for x in range(steps)]

        self.prevStateRewards += [(action, discountedReward) for action, discountedReward in zip(self.prevStates, discountedRewards)]
        self.prevStates = []

        if gameNum % 100 == 0:
            random.shuffle(self.prevStateRewards)

            fmtState = np.array([np.array(state) / 2 for state, reward in self.prevStateRewards])
            fmtState = fmtState.reshape(fmtState.shape + (1,))

            self.valueFunc.train_on_batch(fmtState, np.array([reward for state, reward in self.prevStateRewards]))
            self.prevStateRewards = []

        # Update epsilon
        self.updateEpsilon()

    def buildModel(self, gridDim):
        model = models.Sequential()

        model.add(keras.Input(shape=(gridDim + (1,))))

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())

        model.add(layers.Dense(1000, activation='relu'))
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        return model

    def explore(self, actions):
        return random.choice(actions)

    def optimize(self, grid, actions):
        futures = []

        for action in actions:
            newGrid = grid.createWall(action)
            newGrid = np.array(newGrid) / 2  # Standardize grid for neural network
            newGrid = newGrid.reshape(newGrid.shape + (1,))
            futures += [newGrid]

        actionValues = [(action, value) for action, value in zip(actions, self.valueFunc.predict(np.array(futures)))]

        maxValue = max([actionValue[1] for actionValue in actionValues])
        optActions = [actionValue[0] for actionValue in actionValues if actionValue[1] == maxValue]

        return self.explore(optActions)

    def move(self, grid):
        actions = grid.getValidPlayerMoves()

        if self.train:
            action = self.explore(actions) if random.random() < self.epsilon else self.optimize(grid, actions)
        else:
            action = self.optimize(grid, actions)

        # Save state that is passed through
        newGrid = grid.createWall(action)
        newGrid = np.array(newGrid) / 2  # Standardize grid for neural network
        newGrid = newGrid.reshape(newGrid.shape + (1,))

        self.prevStates += [newGrid]

        return action
