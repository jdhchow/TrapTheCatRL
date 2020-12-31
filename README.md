# TrapTheCatRL
A reinforcement learning algorithm to play Trap the Cat

### Trap the Cat
Trap the Cat is a game played on an NxN board tiled with hexagons. The tiles can
be in one of two states: empty and filled. The game begins with a cat at the
centre. Each turn it moves an empty, adjacent tile. The player's objective is to
prevent the cat from leaving the board. On their turn, they can fill any
empty tile that does not contain the cat. The game ends when the cat reaches the
edge of the board or cannot move.

### Results
The Model/ folder contains some trained models for the player and cat on a 5x5
grid. playerModelSP.py was trained against a cat using a shortest path
algorithm. playerModel_5x5.h5 and catModel_5x5.h5 were trained from scratch
against each other. playerModelSP.py does better than random guessing. It
performs worse against catModel_5x5.h5 than it does against the shortest path
algorithm. This implies catModel_5x5.h5 has learned a better technique than
shortest path.

In future, training on a larger grid would be beneficial to reduce randomness.
There are many opportunities to improve the current algorithms and refactor the
code.
