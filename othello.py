import numpy as np
from copy import deepcopy

def hash_board(s):
    return hash(str(s))

def start_board():
    # 0 for unused, 1 for black, 2 for white
    board = np.zeros((8, 8), dtype=np.int64)
    # initial layout
    board[3,3] = 1
    board[3,4] = 2
    board[4,3] = 2
    board[4,4] = 1
    # black starts
    turn = 1
    return board, turn

def tiles_to_flip(board, turn, x, y):
    ''' Return a list of tiles that would be flipped if given piece was placed at location'''
    tiles = [(x, y)]

    if board[x,y] != 0:
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
            elif board[i, j] == 0:
                done = True
            # if we hit current piece, extend
            elif board[i, j] == turn:
                tiles.extend(potentials)
                done = True
            # else, opponents piece and it can potentially be flipped
            else:
                potentials.append((i, j))
    return tiles

def legal_moves(board, turn):
    moves = []
    # if game end
    if turn <= 0:
        return moves
    for i in range(8):
        for j in range(8):
            # if it would flip tiles over, then legal
            if len(tiles_to_flip(board, turn, i, j)) != 1:
                moves.append((i,j))
    return moves

def flip_turn(turn):
    if turn == 1:
        return 2
    elif turn == 2:
        return 1

def move(board, turn, i, j):
    flip = tiles_to_flip(board, turn, i, j)
    assert len(flip) != 1, 'Illegal move'
    board = deepcopy(board)
    # flip tiles
    for r, c in flip:
        board[r, c] = turn

    turn = flip_turn(turn)
    # if no legal moves, flip
    if len(legal_moves(board, turn)) == 0:
        turn = flip_turn(turn)
        # if still no legal moves, end game
        if len(legal_moves(board, turn)) == 0:
            turn = 0
    return board, turn

def make_current_black(board, turn):
    if turn == 2:
        board = deepcopy(board)
        board[board == 1], board[board == 2] = -2, -1
        board[board == -1], board[board == -2] = 1, 2
        turn = 1
    return board, turn

def get_reward(board, player):
    # assert (board == 0).sum() == 0, 'Reward called when game not done'
    player_count = (board == player).sum()
    opponent_count = (board != player).sum()
    if player_count > opponent_count:
        return 1
    elif player_count < opponent_count:
        return -1
    else:
        return 0
