import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb

class RolloutDataset(Dataset):
    def __init__(self, X, P, V):
        super(RolloutDataset, self).__init__()
        assert len(X) == len(P) and len(P) == len(V), 'Rollouts of different size'
        self.X, self.P, self.V = X, P, V

    def __getitem__(self, index):
        return self.X[index], self.P[index], self.V[index]

    def __len__(self):
        return len(self.X)
