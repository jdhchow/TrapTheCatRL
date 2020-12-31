import datetime
import copy

from Game import Game
from Player import Player
from Cat import RLCat


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


def saveFiles(player, playerModelPath, cat, catModelPath):
    # Save model files if non-deterministic strategy used
    if player.valueFunc is not None:
        player.valueFunc.save(playerModelPath)

    if cat.valueFunc is not None:
        cat.valueFunc.save(catModelPath)


if __name__ == '__main__':
    print(str(datetime.datetime.now()) + ': Started')

    # Set game conditions
    gridDim = (5, 5)  # Coordinates given in the form (y, x)
    initialPercentFill = .6
    simulations = 1000
    playerModelPath = 'Model/playerModel_5x5.h5'
    catModelPath = 'Model/catModel_5x5.h5'
    playerWins = 0
    train = False  # Set to False for testing to prevent updating/saving the model

    # Initialize game, player, and cat
    game = Game(gridDim)
    player = Player()
    cat = RLCat()

    # Build player and cat models (valueFuncPath=None if training from scratch, else load model for training/testing)
    player.newTask(gridDim, valueFuncPath=playerModelPath, train=train)
    cat.newTask(gridDim, valueFuncPath=catModelPath, train=train)

    # Run simulations
    for i in range(1, simulations + 1):
        # Create new game
        game.newGame(initialPercentFill)

        # Reset player history
        player.newGame()
        cat.newGame()

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
        player.updateValueFunc(-game.winner)
        cat.updateValueFunc(game.winner - 1)

        # Update win rate
        playerWins += 1 if game.winner == 0 else 0

        # Display summary (0: Player win, 1: Cat win)
        print('Game ' + str(i) + ' winner is ' + str(game.winner) + ' in ' + str(game.turn) + ' turns')

    # Output simulation summary
    print('Player win rate: ' + str(float(playerWins) / simulations))

    # Save value function
    if train:
        saveFiles(player, playerModelPath, cat, catModelPath)

    print(str(datetime.datetime.now()) + ': Finished')
