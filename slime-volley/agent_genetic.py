from __future__ import annotations
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor, argmax

class BaseAgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.passes = 0
        self.network = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            # left, right, up, nothing
            nn.Linear(100, 4),
            nn.Softmax()
        )
    def forward(self, x: Tensor):
        # x will have dim of 10
        probs = self.network(x)
        self.passes += 1
        return probs

    def reset_state(self):
        self.passes = 0

class GeneticAgentWithGradSensitivity:
    def __init__(self, model: BaseAgentModel):
        self.model : BaseAgentModel = model
        self.elo_score : float = 1000

    def move(self, state : Tensor) -> Tensor:
        logits = self.model.forward(state)
        best_move = argmax(logits, dim=-1)
        logits[best_move].backward()
        return best_move

    def save_model(self, path) -> Path:
        """Save this agent's model parameters and return the saved path."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    @classmethod
    def load_model(cls, path, model_class=BaseAgentModel) -> BaseAgentModel:
        """Load a saved parameter state into a new genetic agent."""

        model = model_class()
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def mutate(
        self,
        mutation_rate: float = 0.01,
        mutation_std: float = 0.1,
        offset_const: float = 0.2
    ) -> "GeneticAgent":
        """Return a child with Bernoulli-masked Gaussian parameter mutations.
        """
        child = GeneticAgent(copy.deepcopy(self.model))
        with torch.no_grad():
            for parameter in child.model.parameters():
                parameter_grad = 0
                if isinstance(parameter.grad, type(None)):
                    parameter_grad = 0
                parameter_grad = parameter.grad.item()
                mutation_mask = torch.bernoulli(
                    torch.full_like(parameter, mutation_rate)
                )
                gaussian_noise = torch.randn_like(parameter) * mutation_std / (((parameter_grad ** 2) + offset_const ) ** 0.5)
                parameter.add_(mutation_mask * gaussian_noise)
        child.model.zero_grad()
        return child


class GeneticAgent:
    def __init__(self, model : BaseAgentModel):
        self.model : BaseAgentModel = model
        self.elo_score = 1000

    def move(self, state: Tensor):
        logits = self.model.forward(state)
        logits.backward()
        return argmax(logits, dim=-1)

    def mutate(
        self,
        mutation_rate: float = 0.01,
        mutation_std: float = 0.1,
    ) -> "GeneticAgent":
        """Return a child with Bernoulli-masked Gaussian parameter mutations.
        """
        child = GeneticAgent(copy.deepcopy(self.model))
        with torch.no_grad():
            for parameter in child.model.parameters():
                mutation_mask = torch.bernoulli(
                    torch.full_like(parameter, mutation_rate)
                )
                gaussian_noise = torch.randn_like(parameter) * mutation_std
                parameter.add_(mutation_mask * gaussian_noise)
        return child