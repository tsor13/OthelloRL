import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pdb

class PolicyValueLoss(nn.Module):
    def __init__(self):
        super(PolicyValueLoss, self).__init__()

    def forward(self, p_hat, v_hat, p, v):
        value_loss = ((v_hat - v)**2).sum()
        # torch.bmm dot product? hopefully
        policy_loss = -torch.bmm(p, torch.log(p_hat)).sum()
        return value_loss + policy_loss

