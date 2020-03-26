import numpy as np
import torch
from torch.nn import functional as F
from params import *

def tensor_from_board(board):
    board = torch.Tensor(board).to(device).unsqueeze(0)
    boards =  torch.cat([board == 1,
                         board == 2])
    boards = boards.float()
    boards = boards.unsqueeze(0)
    return boards
