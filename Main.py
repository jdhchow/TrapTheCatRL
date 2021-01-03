import datetime
import copy
import numpy as np
import random

from Agent import AgentKind
from Game import Game
from Player import RLPlayer
from Cat import RLCat
from Cat import ShortestPathCat


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

A reinforcement learning algorithm to play Trap the Cat

To Do
    -Improve rl algorithms
    -Code refactoring (assuming cat and player models continue to be similar,
     create rl class that player and cat inherit from)
    -Work on scaling up to 11x11
'''

if __name__ == '__main__':
    print(str(datetime.datetime.now()) + ': Started')

    # Set game conditions
    gridDim = (11, 11)  # Coordinates given in the form (y, x)
    simulations = 100000
    playerModelPath = 'Model/playerModel_11x11.h5'
    catModelPath = 'Model/catModel_11x11.h5'
    playerWinsRolling, playerWins = [], 0
    train = True  # Set to False for testing to prevent updating/saving the model

    # Initialize game, player, and cat
    game = Game(gridDim)
    player = RLPlayer(gridDim, valueFuncPath=playerModelPath, train=train)
    # cat = RLCat(gridDim, valueFuncPath=catModelPath, train=train)
    cat = ShortestPathCat(gridDim, valueFuncPath='', train=train)

    # Run simulations
    for i in range(1, simulations + 1):
        # Create new game
        initialPercentFill = random.choice([0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        game.newGame(initialPercentFill)

        # Print simulation information
        # game.grid.displayGrid()

        # Process turns
        while True:
            gridIndex = player.move(copy.deepcopy(game.grid))
            game.movePlayer(gridIndex)

            if game.checkPlayerWin(): break

            direction = cat.move(copy.deepcopy(game.grid))
            game.moveCat(direction)

            if game.checkCatWin(): break

        # Process end of game updates
        player.updateValueFunc(-game.winner.value, i)
        cat.updateValueFunc(game.winner.value - 1, i)

        # Update win rate
        playerWinsRolling += [1 if game.winner == AgentKind.PLAYER else 0]
        playerWins += 1 if game.winner == AgentKind.PLAYER else 0
        if len(playerWinsRolling) > 100: playerWinsRolling = playerWinsRolling[1:]

        # Display summary (0: Player win, 1: Cat win)
        winner = 'Cat' if game.winner == AgentKind.CAT else 'Player'
        print('Game ' + str(i) + ' : ' + winner + ' Won : ' + str(game.turn) + ' Turns : Player Win Rate ' + str(np.mean(playerWinsRolling)))

        # Save value function every 1000 games
        if train and i % 1000 == 0:
            player.save()
            cat.save()
            print('Epsilon is: ' + str(player.epsilon))  # In case we want to restart training without resetting threshold

    # Print cumulative player win rate
    print('Player Win Rate ' + str(float(playerWins) / simulations))

    if train:
        player.save()
        cat.save()

    print(str(datetime.datetime.now()) + ': Finished')
