from __future__ import annotations

import copy
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

OBSERVATION_SIZE = 12
INITIAL_RATING = 1000.0
ACTION_COUNT = 4


class BaseAgentModel(nn.Module):
    def __init__(self, hidden_size: int = 100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_COUNT),
        )

    def forward(self, normalized_observation: Tensor) -> Tensor:
        return self.network(normalized_observation)


_agent_id_counter = itertools.count()


def reserve_agent_ids_through(highest_used_agent_id: int):
    global _agent_id_counter
    next_free_agent_id = max(next(_agent_id_counter), highest_used_agent_id + 1)
    _agent_id_counter = itertools.count(next_free_agent_id)


class GeneticAgent:
    def __init__(self, model: BaseAgentModel | None = None):
        self.model: BaseAgentModel = model if model is not None else BaseAgentModel()
        self.rating: float = INITIAL_RATING
        self.agent_id: int = next(_agent_id_counter)

    def move(self, observation: Tensor) -> Tensor:
        action_logits = self.model(observation)
        return torch.argmax(action_logits, dim=-1)

    def mutate(self, mutation_std: float = 0.02) -> GeneticAgent:
        child = GeneticAgent(copy.deepcopy(self.model))
        with torch.no_grad():
            for parameter in child.model.parameters():
                parameter.add_(torch.randn_like(parameter) * mutation_std)
        return child

    def weights(self) -> dict[str, Tensor]:
        return self.model.state_dict()

    @classmethod
    def from_weights(cls, model_weights: dict[str, Tensor]) -> GeneticAgent:
        model = BaseAgentModel()
        model.load_state_dict(model_weights)
        return cls(model)

    def save_model(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.weights(), path)
        return path

    @classmethod
    def load_model(cls, path) -> GeneticAgent:
        return cls.from_weights(torch.load(path, map_location="cpu"))
