from typing import Dict, List, Sequence, Tuple

import numpy as np
from threadpoolctl import threadpool_limits
from torch import Tensor

from pufferlib.ocean.slimevolley.slimevolley import SlimeVolley as PufferSlimeVolley

AGENTS_PER_MATCH = 2
LEFT_SLOT = 0
MATH_THREADS_PER_PROCESS = 1

WIN = 1.0
DRAW = 0.5
POINTS_TO_WIN = 3
LEFT_WON_POINT_REWARD = 1.0
RIGHT_WON_POINT_REWARD = -1.0

NOTHING_ACTION_INDEX = 3
FORWARD_BACKWARD_JUMP_BY_ACTION_INDEX = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
)

Pairing = Tuple[int, int]
ModelWeights = Dict[str, Tensor]
NumpyLayer = Tuple[np.ndarray, np.ndarray]


def _numpy_layers(model_weights: ModelWeights) -> List[NumpyLayer]:
    parameter_names = list(model_weights.keys())
    return [
        (
            model_weights[weight_name].detach().numpy().astype(np.float32).T.copy(),
            model_weights[bias_name].detach().numpy().astype(np.float32),
        )
        for weight_name, bias_name in zip(parameter_names[0::2], parameter_names[1::2])
    ]


def _action_logits(layers: List[NumpyLayer], observations: np.ndarray) -> np.ndarray:
    activations = observations
    for weights, biases in layers[:-1]:
        activations = np.maximum(activations @ weights + biases, 0.0)
    output_weights, output_biases = layers[-1]
    return activations @ output_weights + output_biases


class PolicyGroups:
    def __init__(self, policy_weights: Sequence[ModelWeights], policy_index_by_slot: Sequence[int]):
        self.layers_by_policy = [_numpy_layers(model_weights) for model_weights in policy_weights]
        self.policy_index_by_slot = np.asarray(policy_index_by_slot)
        self.groups: List[Tuple[int, np.ndarray]] = []
        self.set_active_slots(np.arange(len(self.policy_index_by_slot)))

    def set_active_slots(self, active_slots: np.ndarray):
        if len(active_slots) == 0:
            self.groups = []
            return
        slots_by_policy = active_slots[np.argsort(self.policy_index_by_slot[active_slots], kind="stable")]
        policy_of_sorted_slot = self.policy_index_by_slot[slots_by_policy]
        policy_changes = np.flatnonzero(policy_of_sorted_slot[1:] != policy_of_sorted_slot[:-1]) + 1
        group_starts = np.concatenate(([0], policy_changes))
        group_ends = np.concatenate((policy_changes, [len(slots_by_policy)]))
        self.groups = [
            (int(policy_of_sorted_slot[start]), slots_by_policy[start:end])
            for start, end in zip(group_starts, group_ends)
        ]

    def fill_action_indices(self, observations: np.ndarray, action_indices: np.ndarray):
        for policy_index, slots in self.groups:
            action_indices[slots] = _action_logits(self.layers_by_policy[policy_index], observations[slots]).argmax(axis=1)


def _left_share_of_points(left_points_won: int, right_points_won: int) -> float:
    total_points = left_points_won + right_points_won
    if total_points == 0:
        return DRAW
    return left_points_won / total_points


def _slots_of_unfinished_matches(match_finished: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.repeat(~match_finished, AGENTS_PER_MATCH))


def play_matches(
    policy_weights: Sequence[ModelWeights],
    pairings: Sequence[Pairing],
    seed: int = 0,
    points_to_win: int = POINTS_TO_WIN,
) -> List[float]:
    with threadpool_limits(limits=MATH_THREADS_PER_PROCESS):
        return _play_matches_single_threaded(policy_weights, pairings, seed, points_to_win)


def _play_matches_single_threaded(
    policy_weights: Sequence[ModelWeights],
    pairings: Sequence[Pairing],
    seed: int,
    points_to_win: int,
) -> List[float]:
    policy_index_by_slot = [policy_index for pairing in pairings for policy_index in pairing]
    policies = PolicyGroups(policy_weights, policy_index_by_slot)
    environment = PufferSlimeVolley(num_envs=len(pairings), num_agents=AGENTS_PER_MATCH)
    observations, _ = environment.reset(seed=seed)

    action_indices = np.full(len(policy_index_by_slot), NOTHING_ACTION_INDEX)
    left_points_won = np.zeros(len(pairings), dtype=int)
    right_points_won = np.zeros(len(pairings), dtype=int)
    match_finished = np.zeros(len(pairings), dtype=bool)
    while not match_finished.all():
        policies.fill_action_indices(observations, action_indices)
        _, rewards, terminals, _, _ = environment.step(FORWARD_BACKWARD_JUMP_BY_ACTION_INDEX[action_indices])
        still_playing = ~match_finished
        left_rewards = rewards[LEFT_SLOT::AGENTS_PER_MATCH]
        left_points_won += still_playing & (left_rewards == LEFT_WON_POINT_REWARD)
        right_points_won += still_playing & (left_rewards == RIGHT_WON_POINT_REWARD)
        someone_reached_target = np.maximum(left_points_won, right_points_won) >= points_to_win
        episode_ended = terminals[LEFT_SLOT::AGENTS_PER_MATCH].astype(bool)
        newly_finished = still_playing & (someone_reached_target | episode_ended)
        if newly_finished.any():
            match_finished |= newly_finished
            action_indices[np.repeat(match_finished, AGENTS_PER_MATCH)] = NOTHING_ACTION_INDEX
            policies.set_active_slots(_slots_of_unfinished_matches(match_finished))
    environment.close()
    return [_left_share_of_points(left, right) for left, right in zip(left_points_won, right_points_won)]
