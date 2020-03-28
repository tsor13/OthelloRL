import torch
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from MCTS import MCTS
from OthelloZero import OthelloZero
from PolicyValueLoss import PolicyValueLoss
from RolloutDataset import RolloutDataset
from helper import *
from tqdm import tqdm
import pdb

def train(params={}):
    model = params.get('model', OthelloZero(res_layers=4, channels=10))
    iterations = params.get('iterations', 1000)
    max_time = params.get('max time', 1e6)
    num_tree_searches = params.get('num tree searches', 10)

    batch_size = params.get('batch size', 8)
    epochs = params.get('epochs', 2)

    loss_function = params.get('loss function', PolicyValueLoss())

    for i in range(range_tree_searches):
        # do tree search
        nodes, boards, actions = MCTCS(model,
                                       iterations = iterations,
                                       max_time = max_time)
        # convert to goals
        X, P, V = get_objectives(nodes, boards, actions)
        # augment
        X, P, V = augment(X, P, V)

        # training loaders
        train_dataset = RolloutDataset(X, P, V)
        train_loader = DataLoader(train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 8)

        # optimizer
        optimizer = optim.Adam(model.parameters(), learning_rate)

        # training loop
        loop = tqdm(total = len(train_loader)*epochs, position=0, leave=False)
        model.train()
        for epoch in range(epochs):
            for batch, (x, p, v) in enumerate(train_loader):
                x, p, v = x.to(device), p.to(device), v.to(device)

                optimizer.zero_grad()

                p_hat, v_hat = model(x)

                loss = loss_function(p_hat, v_hat, p, v)
                loss.backward()

                optimizer.step()

                loop.update(1)
                loop.set_description('loss:{:.3f}'.format(loss.item()))
