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
