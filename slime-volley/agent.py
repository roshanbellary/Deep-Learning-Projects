from model import QTrainer, QNet
import random 
import numpy as np
import torch
import math
from collections import deque

from game import GameState
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, type):
        self.epsilon = 0.5
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.n_games = 0

    def get_action(self, state): 
        # state constists of 10 dimensions
        # ball position and ball velocity
        # player position and player vertical velocity
        # opponent position and opponent vertical velocity
        #action consists of 4 possibilities: move left, right, jump, stay
        self.epsilon =  0.2 * math.exp(math.log(0.5)/100 * self.n_games)

        final_move = [0, 0, 0, 0]

        if (random.random() > self.epsilon):
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)

            move = torch.argmax(prediction).item()

            final_move[move] = 1
        else:
            move = random.randint(0, 3)
            final_move[move] = 1

        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory 
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) 



if __name__ == "__main__":
    game = GameState()
    agent = Agent(game)
    agent.train(manual_control=False)