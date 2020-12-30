import datetime
from Game import Game
from Player import Player


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

An RL algorithm to play trap the cat

To Do:
    -Intelligent moves for cat (currently picks first available move)
    -Options for higher dimensions and/or multiple cats with player able to flip multiple tiles per round
'''


if __name__ == '__main__':
    print(str(datetime.datetime.now()) + ': Started')

    # Set game conditions
    gridDim = (5, 5)  # Coordinates given in the form (y, x)
    initialPercentFill = .6
    simulations = 100
    modelPath = 'Model/initialModel.h5'
    wins = 0

    # Initialize game and player
    game = Game()
    player = Player()

    # Build player model
    player.newTask(gridDim)

    # Run simulations
    for i in range(1, simulations + 1):
        # Create new game
        game.newGame(*gridDim, initialPercentFill)

        # Save number of turns
        turns = 0

        # Reset player history
        player.newEpoch()

        # Print simulation information
        print('Game ' + str(i))
        # game.displayGrid()

        # Process turns
        while game.winner == 1:
            gridIndex = player.move(game.grid)
            game.createWall(gridIndex)
            game.move()
            turns += 1

        player.updateValueFunc(game.winner)
        print('Game winner ' + str(game.winner) + ' in ' + str(turns) + ' turns')

        wins = wins + 1 if game.winner == 0 else wins

    print(float(wins) / simulations)

    # Save value function
    player.valueFunc.save(modelPath)

    print(str(datetime.datetime.now()) + ': Finished')
