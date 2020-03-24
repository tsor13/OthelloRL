import numpy as np
from environment import OthelloEnv

def get_char(p):
    if p == 0:
        return '-'
    elif p == 1:
        return 'X'
    elif p == 2:
        return 'O'

def print_board(board):
    print('  0 1 2 3 4 5 6 7')
    for i, row in enumerate(board):
        s = str(i) + ' '+ ' '.join(list(map(get_char, row.tolist())))
        print(s)

if __name__ == '__main__':
    oe = OthelloEnv()
    turn, board, moves = oe.state()
    while turn != 0:
        print_board(board)
        valid = False
        while not valid:
            move = tuple(map(int, input(get_char(turn) + ': ').strip().split()))
            if len(move) == 2 and move in moves:
                try:
                    turn, board, moves = oe.move(*move)
                    valid = True
                except:
                    print('Not Valid Move')
                    valid = False
    print_board(board)
