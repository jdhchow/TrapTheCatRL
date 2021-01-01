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
    gridDim = (11, 11)  # Coordinates given in the form (y, x)
    initialPercentFill = 0.15
    simulations = 10000
    playerModelPath = 'Model/playerModel_11x11.h5'
    catModelPath = 'Model/catModel_11x11.h5'
    playerWins = 0
    train = True  # Set to False for testing to prevent updating/saving the model

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
        player.updateValueFunc(-game.winner, i)
        cat.updateValueFunc(game.winner - 1, i)

        # Update win rate
        playerWins += 1 if game.winner == 0 else 0

        # Display summary (0: Player win, 1: Cat win)
        winner = 'cat' if game.winner == 1 else 'player'
        print('Game ' + str(i) + ' winner is ' + winner + ' in ' + str(game.turn) + ' turns')

        # Save value function every 100 games
        if train and i % 100 == 0:
            saveFiles(player, playerModelPath, cat, catModelPath)
            print('Epsilon is: ' + str(player.epsilon))  # In case we want to restart training without resetting threshold

    # Output simulation summary
    print('Player win rate: ' + str(float(playerWins) / simulations))

    print(str(datetime.datetime.now()) + ': Finished')
