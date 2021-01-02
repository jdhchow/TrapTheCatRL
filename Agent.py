from enum import Enum 
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
import random

class AgentKind(Enum):
    PLAYER = 0
    CAT = 1

    def __int__(self):
        if self == PLAYER:
            return 0
        if self == CAT:
            return 1
        raise NotImplementedError

class Agent(object):
    def __init__(self, gridDim, valueFuncPath, train):
        self.gridDim = gridDim
        self.valueFuncPath = valueFuncPath
        self.train = train

    def updateValueFunc(self, terminalReward, gameNum):
        raise NotImplementedError

    def move(self, grid):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError

    def validActions(self, grid):
        raise NotImplementedError

    def applyAction(self, grid, action):
        raise NotImplementedError

    @property
    def kind(self):
        raise NotImplementedError

class RLAgent(Agent):
    def __init__(self, gridDim, valueFuncPath, train):
        super().__init__(gridDim, valueFuncPath, train)

        self.epsilon = 0.5
        self.epsilonStepSize = 0.001
        self.gamma = 0.8

        # Set for each task
        self.actions = None

        # Set for each game
        self.prevStates = []

        if os.path.exists(valueFuncPath):
            self.valueFunc = keras.models.load_model(valueFuncPath)
        else:
            self.valueFunc = self.buildModel(gridDim)

    def save(self):
        self.valueFunc.save(self.valueFuncPath)

    def updateEpsilon(self):
        self.epsilon = 1 / (1 / self.epsilon + self.epsilonStepSize)

    def updateValueFunc(self, terminalReward, gameNum):
        if not self.train:
            return

        # If game does not end without cat moving
        if self.prevStates:
            steps = len(self.prevStates)
            discountedRewards = np.array([(self.gamma ** (steps - x)) * terminalReward for x in range(steps)])

            fmtStates = np.array([np.array(state) / 2 for state in self.prevStates])
            fmtStates = fmtStates.reshape(fmtStates.shape + (1,))

            self.valueFunc.train_on_batch(fmtStates, discountedRewards)
            self.prevStates = []

        # Update epsilon
        self.updateEpsilon()

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
        futures = []

        for action in actions:
            newGrid = self.applyAction(grid, action)
            newGrid = np.array(newGrid) / 2  # Standardize grid for neural network
            newGrid = newGrid.reshape(newGrid.shape + (1,))
            futures += [newGrid]

        actionValues = [(action, value) for action, value in zip(actions,  self.valueFunc.predict(np.array(futures)))]

        maxValue = max([actionValue[1] for actionValue in actionValues])
        optActions = [actionValue[0] for actionValue in actionValues if actionValue[1] == maxValue]

        return self.explore(optActions)

    def move(self, grid):
        actions = self.validActions(grid)

        if self.train:
            action = self.explore(actions) if random.random() < self.epsilon else self.optimize(grid, actions)
        else:
            action = self.optimize(grid, actions)

        # Save state that is passed through
        newGrid = self.applyAction(grid, action)
        newGrid = np.array(newGrid) / 2  # Standardize grid for neural network
        newGrid = newGrid.reshape(newGrid.shape + (1,))

        self.prevStates += [newGrid]

        return action