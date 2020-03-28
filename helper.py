import numpy as np
import torch
from torch.nn import functional as F
from params import *
import pdb

def tensor_from_board(board, actions):
    board = torch.Tensor(board).to(device).unsqueeze(0)
    x =  torch.cat([(board == 1).float(),
                    (board == 2).float(),
                    torch.zeros(board.shape)])
    x = x.float()
    for i, j in actions:
        x[2,i,j] = 1
    return x

def get_objectives(nodes, boards, actions):
    # policy = normalized counts
    # value = mean value
    X, P, V = [], [], []
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
    return X, P, V

def augment(X, P, V):
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
    return X_, P_, V_
