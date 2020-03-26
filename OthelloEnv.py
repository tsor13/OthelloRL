import numpy as np

class OthelloEnv:
    def __init__(self):
        super(OthelloEnv).__init__()
        # 0 for unused, 1 for black, 2 for white
        self.board = np.zeros((8, 8))
        # initial layout
        self.board[3,3] = 1
        self.board[3,4] = 2
        self.board[4,3] = 2
        self.board[4,4] = 1
        self.turn = 1

    def tiles_to_flip(self, x, y):
        ''' Return a list of tiles that would be flipped if given piece was placed at location'''
        tiles = []

        if self.board[x,y] != 0:
            return tiles

        def in_bounds(i, j):
            return not (i < 0 or j < 0 or i >= 8 or j >= 8)

        # define search directions
        directions = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        # look in each direction
        for x_d, y_d in directions:
            potentials = []
            i, j = x, y
            done = False
            while not done:
                i, j = i+x_d, j+y_d
                # if out of bounds, never hits own piece
                if not in_bounds(i, j):
                    done = True
                # if hits empty slot, return
                elif self.board[i, j] == 0:
                    done = True
                # if we hit current piece, extend
                elif self.board[i, j] == self.turn:
                    tiles.extend(potentials)
                    done = True
                # else, opponents piece and it can potentially be flipped
                else:
                    potentials.append((i, j))
        return tiles


    def legal_moves(self):
        moves = []
        # if game end
        if self.turn == 0:
            return moves
        for i in range(8):
            for j in range(8):
                # if it would flip tiles over, then legal
                if len(self.tiles_to_flip(i,j)):
                    moves.append((i,j))
        return moves


    def state(self):
        ''' Returns a tuple of state, in the following order:
            turn - 1 (black), 2 (white), 0 (game end)
            board - self.board (8x8 numpy array)
            legal moves - list of tuples of legal move positions '''
        return (self.turn, self.board, self.legal_moves())

    def flip_turn(self):
        if self.turn == 1:
            self.turn = 2
        elif self.turn == 2:
            self.turn = 1

    def move(self, i, j):
        flip = self.tiles_to_flip(i, j)
        assert len(flip) != 0, 'Illegal move'
        # place tile
        self.board[i,j] = self.turn
        for r, c in flip:
            self.board[r, c] = self.turn

        self.flip_turn()
        # if no legal moves, flip
        if len(self.legal_moves()) == 0:
            self.flip_turn()
            # if still no legal moves, end game
            if len(self.legal_moves()) == 0:
                self.turn = 0
        return self.state()

if __name__ == '__main__':
    oe = OthelloEnv()
