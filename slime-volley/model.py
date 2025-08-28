import torch
import torch.nn as nn
import torch.optim as otpim
import torch.nn.functional as F
import os 

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4) # move left, move right, jump, stay
        )
    def forward(self, x):
        if x.ndim == 1:
            x = torch.unsqueeze(x, 0)
        x = self.flatten(x)
        logits = self.stack(x)
        return logits 
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma 
        self.model = model 
        self.optimizer = otpim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if (len(state.shape) == 1):
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

