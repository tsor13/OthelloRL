from othello import *
from helper import *
from copy import deepcopy
from math import sqrt
from tqdm import tqdm
import time
import pdb

def MCTS(model, iterations = 10, C = 100, max_time = 1e6):
    # N, W, Q, P
    # key: str(move_history), action (tuple)
    nodes = {}
    # boards (for backprop)
    # key: str(move_history)
    boards = {}
    # actions
    # key: str(move_history)
    actions = {}

    # start off with no move hist
    move_history = []
    board, turn = start_board()

    boards[str(move_history)] = board

    i = 0
    start = time.time()
    done = False
    loop = tqdm(total = 100, position=0, leave=False)
    last_perc = 0
    while not done:
        move_history = []
        board, turn = start_board()
        # keep track of whose turn it was
        turns = [1]
        # while game not done
        while turn != 0:
            # populate actions if unexplored yet
            if not actions.get(str(move_history), []):
                actions[str(move_history)] = legal_moves(board, 1)
                x = tensor_from_board(board, actions[str(move_history)])
                x = x.unsqueeze(0)
                p, v = model(x)
                for a in actions[str(move_history)]:
                    i, j = a
                    nodes[str(move_history),a] = [0, 0, 0, p[0, i, j].item()]

            # get total
            total_N = 0
            for a in actions[str(move_history)]:
                N, W, Q, P = nodes[str(move_history), a]
                total_N += N

            # find next action to explore
            max_val, max_action = 0, None
            for a in actions[str(move_history)]:
                N, W, Q, P = nodes[str(move_history), a]
                # choose nodes with high Q or high P/low N
                val = Q + C * P * sqrt((total_N+.1)/(N+.1))
                if val > max_val:
                    max_val = val
                    max_action = a

            # take move
            board, turn = move(board, 1, max_action[0], max_action[1])
            move_history.append(max_action)

            if turn != 0:
                turns.append(turn)
            board, turn = make_current_black(board, turn)
            boards[str(move_history)] = board

            i += 1
            perc = max(i/iterations, (time.time() - start)/max_time)
            perc = 100*round(perc, 2)
            while last_perc < perc:
                last_perc += 1
                loop.update(1)
            if i > iterations or time.time() - start > max_time:
                done = True

        # remove last turn (end)
        turns.pop()
        # get reward with respect to last player
        v = get_reward(board, turns[-1])
        while move_history:
            a = move_history.pop()
            # N += 1
            nodes[str(move_history), a][0] += 1
            # W += v
            nodes[str(move_history), a][1] += v
            # Q = (N+W) / 2 / N
            N, W, _, _ = nodes[str(move_history), a]
            nodes[str(move_history), a][2] = (N+W) / 2 / N
            # if turn changes, the reward changes
            try:
                t = turns.pop()
                if t == 2:
                    v *= -1
            except:
                pass

    return nodes, boards, actions


if __name__ == '__main__':
    from OthelloZero import OthelloZero
    model = OthelloZero(res_layers=4, channels=10)
    print('start')
    nodes, boards, actions = MCTS(model, 20)
    print('end')
    X, P, V = get_objectives(nodes, boards, actions)
    X, P, V = augment(X, P, V)
