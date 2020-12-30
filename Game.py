import random


'''
Author: Jonathan Chow
Date Created: 2020-12-29
Python Version: 3.7

Class for the game
'''

class Game:
    def __init__(self):
        self.grid = None
        self.catLoc = None
        self.gridDim = None
        self.winner = None

    def createWall(self, index):
        col = index // self.gridDim[0]
        row = index - self.gridDim[0] * (index // self.gridDim[0])

        # Check collision with cat
        if self.grid[col][row] != 2:
            self.grid[col][row] = 1

    def newGame(self, y, x, pFill):
        # (Re)set the winner
        self.winner = 1

        # Generate initial grid
        self.grid = [[0 for row in range(x)] for col in range(y)]
        self.gridDim = (y, x)

        # Flip random, unoccupied tiles
        totalTiles = y * x
        nFill = int(totalTiles * pFill)
        fillIndices = random.sample(population=range(totalTiles), k=nFill)

        for index in fillIndices:
            self.createWall(index)

        # Place cat
        self.catLoc = (y // 2, x // 2)
        self.grid[self.catLoc[0]][self.catLoc[1]] = 2

    def displayGrid(self):
        for col in range(self.gridDim[0]):
            offset = '' if col % 2 == 0 else ' '
            print(offset + ' '.join([str(tile) for tile in self.grid[col]]))

    def is_winning_position(self, pos):
        # Check if cat is on edge of grid
        for i in range(2):
            if not {0, self.gridDim[i]-1}.isdisjoint([pos[i]]):
                return True
        return False

    def has_cat_won(self):
        return self.is_winning_position(self.catLoc)

    def get_valid_moves(self, pos):
        # These tuples are all (y,x)
        offsets = [-1 + pos[0] % 2, 0 + pos[0] % 2]
        hardOffset = [(0, -1), (0, 1)] + [(y, x) for y in [-1, 1] for x in offsets]
        potentialMoves = [tuple(sum(joined) for joined in zip(pos, offset)) for offset in hardOffset]

        # Remove blocked moves
        return [moveTuple for moveTuple in potentialMoves if self.grid[moveTuple[0]][moveTuple[1]] != 1]

    def get_dist(self, pos):
        q = [(pos, 0)]
        seen = set()
        while q:
            curr, dist = q.pop(0)
            if curr in seen:
                continue
            seen.add(curr)
            if self.is_winning_position(curr):
                return dist
            q.extend([(next_, dist+1) for next_ in self.get_valid_moves(curr)])

        # We can't reach the edge anymore :(
        return 10000

    def move(self):
        # Check win by cat
        if self.has_cat_won():
            self.winner = -1
            return

        potentialMoves = [x[1] for x in sorted([(self.get_dist(x), x) for x in self.get_valid_moves(self.catLoc)])]


        # Check player win
        if not potentialMoves:
            self.winner = 0
            return

        # Select move
        move = potentialMoves[0]

        # Update grid and cat location
        self.grid[self.catLoc[0]][self.catLoc[1]] = 0
        self.grid[move[0]][move[1]] = 2
        self.catLoc = move
