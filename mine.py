import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from scipy.special import logsumexp
import math

class MINE():
    def __init__(self, architecture, data, device = 'cuda', batch_size = 512):
        self.device = device
        self.model = architecture().to(self.device)
        self.data = data
        self.batch_size = batch_size
        self.loader = DataLoader(self.data, shuffle = True, batch_size = self.batch_size)

    def optimize(self, epochs = 100, history = True, verbose = True):
        if history:
            bounds, joint_scores, product_scores = [], [], []
        
        lr = 1e-3
        optimizer = Adam(self.model.parameters(), lr = lr)

        for epoch in range(epochs):
            if verbose:
                print(f'Epoch = {epoch}')
            
            for batch in self.loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                z = y[torch.randperm(y.shape[0])]

                optimizer.zero_grad()

                joint_score = torch.mean(self.model(x, y))
                product_score = (torch.logsumexp(self.model(x, z), 0) - math.log(x.shape[0]))[0]
                bound = -joint_score + product_score + product_score**2

                if history:
                    bounds.append(-bound.cpu().detach().numpy())
                    joint_scores.append(joint_score.cpu().detach().numpy())
                    product_scores.append(product_score.cpu().detach().numpy())

                bound.backward()
                optimizer.step()

        if history:
            return self.model, bounds, joint_scores, product_scores
        return self.model

    def estimate(self):
        joint_scores, product_scores = [], []
        
        for batch in self.loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            z = y[torch.randperm(y.shape[0])]
            
            with torch.no_grad():
                joint_score = self.model(x, y)
                product_score = self.model(x, z)

            joint_scores.extend(joint_score.cpu().detach().numpy())
            product_scores.extend(product_score.cpu().detach().numpy())

        bound = np.mean(joint_scores) - (logsumexp(product_scores) - np.log(len(product_scores)))
        return bound
