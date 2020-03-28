from othello import *
from helper import *
from copy import deepcopy
from math import sqrt
import pdb

def MCTS(model, iterations = 10):
    # explore param
    C = 100

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

    for i in range(iterations):
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

def get_objectives(nodes, boards, actions):
    # policy = normalized counts
    # value = mean value
    X, V, P = [], [], []
    for key, board in boards.items():
        total_n = 0
        total_value = 0
        if key in actions:
            for a in actions[key]:
                N, W, _, _ = nodes[key, a]
                total_n += N
                total_value += W
            v = total_value / total_n
            p = np.zeros(board.shape)
            for a in actions[key]:
                N, _, _, _ = nodes[key, a]
                i, j = a
                p[i,j] = N / total_n
            # get tensor
            x = tensor_from_board(board, actions[key])
            x = x.numpy()
            # add to arrays
            X.append(x)
            V.append(v)
            P.append(p)
    return X, V, P

def augment(X, V, P):
    # horizontal flip, vertical flip, transpose
    # these can all be transposed with eachother
    X_, V_, P_ = [], [], []
    def horizontal_flip(x, v, p):
        return x[:,::-1], v, p[::-1]
    def vertical_flip(x, v, p):
        return x[:,:,::-1], v, p[:,::-1]
    def transpose(x, v, p):
        # TODO correct?
        return np.transpose(x, (0,2,1)), v, p.T
    def add(x, v, p):
        X_.append(x)
        V_.append(v)
        P_.append(p)
    for obs in zip(X, V, P):
        add(*horizontal_flip(*obs))
        add(*vertical_flip(*obs))
        add(*transpose(*obs))
        add(*obs)
    return X_, V_, P_

if __name__ == '__main__':
    from OthelloZero import OthelloZero
    model = OthelloZero(res_layers=4, channels=10)
    print('start')
    nodes, boards, actions = MCTS(model, 10)
    X, V, P = get_objectives(nodes, boards, actions)
    X, V, P = augment(X, V, P)
    print('end')
