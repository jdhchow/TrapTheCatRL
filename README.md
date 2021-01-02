# TrapTheCatRL
A reinforcement learning algorithm to play Trap the Cat

### Trap the Cat
Trap the Cat is a game played on an NxN board tiled with hexagons. The tiles can
be in one of two states: empty and filled. The game begins with a cat at the
centre. Each turn it moves an empty, adjacent tile. The player's objective is to
prevent the cat from leaving the board. On their turn, they can fill any
empty tile that does not contain the cat. The game ends when the cat reaches the
edge of the board or cannot move.

### Reinforcement Learning
The player and cat know the state that will result from their actions. The
state-value function is represented by a convolutional neural network (CNN).
First-visit Monte Carlo updates are applied after each game. An epsilon
is set and decreased during training which acts as a threshold for exploration
versus optimization.

### Results
playerModel_11x11_SP.h5 is a model for the player trained against a cat
following the shortest path algorithm. There are many opportunities to improve
the current algorithms.
