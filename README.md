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
Player.py and Cat.py contain reinforcement learning algorithms for the player
and cat respectively. Cat.py also contains other deterministic strategies.
