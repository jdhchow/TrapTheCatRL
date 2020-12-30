import datetime
import copy
from Game import Game
from Player import Player
from Cat import Cat


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

A reinforcement learning algorithm to play Trap the Cat

To Do:
    -Options for higher dimensions, multiple cats, and/or multiple players
'''


if __name__ == '__main__':
    print(str(datetime.datetime.now()) + ': Started')

    # Set game conditions
    gridDim = (5, 5)  # Coordinates given in the form (y, x)
    initialPercentFill = .6
    simulations = 100
    playerModelPath = 'Model/playerModel.h5'
    catModelPath = 'Model/catModel.h5'
    playerWins = 0

    # Initialize game, player, and cat
    game = Game(gridDim)
    player = Player()
    cat = Cat()

    # Build player and cat models (add playerModelPath to parameters to use existing model)
    player.newTask(gridDim)
    cat.newTask(gridDim)

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
            gridIndex = player.move(copy.deepcopy(game.grid.grid))
            game.movePlayer(gridIndex)

            if game.checkPlayerWin(): break

            direction = cat.move(copy.deepcopy(game.grid.grid), copy.deepcopy(game.grid.catLoc))
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
    player.valueFunc.save(playerModelPath)
    cat.valueFunc.save(catModelPath)

    print(str(datetime.datetime.now()) + ': Finished')
