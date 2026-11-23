from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from threadpoolctl import threadpool_limits
from torch import Tensor

from pufferlib.ocean.slimevolley.slimevolley import SlimeVolley as PufferSlimeVolley

try:
    from . import kinematic_bot
except ImportError:
    import kinematic_bot

AGENTS_PER_MATCH = 2
LEFT_SLOT = 0
MATH_THREADS_PER_PROCESS = 1

WIN = 1.0
DRAW = 0.5
POINTS_TO_WIN = 3
LEFT_WON_POINT_REWARD = 1.0
RIGHT_WON_POINT_REWARD = -1.0
RETURN_OVER_NET_REWARD = 0.01
KINEMATIC_BOT = "kinematic_bot"
PUFFERLIB_BOT = "pufferlib_bot"
NO_SIDE = 0

PUFFERLIB_BOT_OWN_X_WEIGHT = -23.757145
PUFFERLIB_BOT_BALL_X_WEIGHT = 23.206863
PUFFERLIB_BOT_BALL_VX_WEIGHT = 0.7943352
PUFFERLIB_BOT_BACKWARD_BIAS = 1.4617119
PUFFERLIB_BOT_FORWARD_FROM_BACKWARD = -64.6463748
PUFFERLIB_BOT_FORWARD_BIAS = 22.4668393
BUTTON_COUNT = 3
FORWARD_BUTTON = 0
BACKWARD_BUTTON = 1
JUMP_BUTTON = 2

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
Policy = Union[ModelWeights, str]
NumpyLayer = Tuple[np.ndarray, np.ndarray]
ButtonPolicy = Callable[[np.ndarray], np.ndarray]


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


def pufferlib_bot_buttons(observations: np.ndarray) -> np.ndarray:
    backward_drive = (
        PUFFERLIB_BOT_OWN_X_WEIGHT * observations[:, kinematic_bot.OWN_X_INDEX]
        + PUFFERLIB_BOT_BALL_X_WEIGHT * observations[:, kinematic_bot.BALL_X_INDEX]
        + PUFFERLIB_BOT_BALL_VX_WEIGHT * observations[:, kinematic_bot.BALL_VX_INDEX]
        + PUFFERLIB_BOT_BACKWARD_BIAS
    )
    forward_drive = PUFFERLIB_BOT_FORWARD_FROM_BACKWARD * backward_drive + PUFFERLIB_BOT_FORWARD_BIAS
    buttons = np.zeros((len(observations), BUTTON_COUNT), dtype=int)
    buttons[:, FORWARD_BUTTON] = forward_drive > 0
    buttons[:, BACKWARD_BUTTON] = backward_drive > 0
    buttons[:, JUMP_BUTTON] = 1
    return buttons


def kinematic_bot_buttons(observations: np.ndarray) -> np.ndarray:
    return FORWARD_BACKWARD_JUMP_BY_ACTION_INDEX[kinematic_bot.choose_actions(observations)]


SCRIPTED_BUTTON_POLICIES: Dict[str, ButtonPolicy] = {
    KINEMATIC_BOT: kinematic_bot_buttons,
    PUFFERLIB_BOT: pufferlib_bot_buttons,
}


def _network_button_policy(model_weights: ModelWeights) -> ButtonPolicy:
    layers = _numpy_layers(model_weights)
    return lambda observations: FORWARD_BACKWARD_JUMP_BY_ACTION_INDEX[_action_logits(layers, observations).argmax(axis=1)]


def _button_policy(policy: Policy) -> ButtonPolicy:
    if isinstance(policy, str):
        if policy not in SCRIPTED_BUTTON_POLICIES:
            raise ValueError(f"Unknown scripted policy: {policy!r}")
        return SCRIPTED_BUTTON_POLICIES[policy]
    return _network_button_policy(policy)


class PolicyGroups:
    def __init__(self, policy_weights: Sequence[Policy], policy_index_by_slot: Sequence[int]):
        self.button_policies = [_button_policy(policy) for policy in policy_weights]
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

    def fill_buttons(self, observations: np.ndarray, buttons: np.ndarray):
        for policy_index, slots in self.groups:
            buttons[slots] = self.button_policies[policy_index](observations[slots])


def _left_share_of_reward(left_reward: float, right_reward: float) -> float:
    total_reward = left_reward + right_reward
    if total_reward == 0:
        return DRAW
    return left_reward / total_reward


def _ball_side_seen_from_left(observations: np.ndarray) -> np.ndarray:
    return np.sign(observations[LEFT_SLOT::AGENTS_PER_MATCH, kinematic_bot.BALL_X_INDEX]).astype(int)


def _slots_of_unfinished_matches(match_finished: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.repeat(~match_finished, AGENTS_PER_MATCH))


def play_matches(
    policy_weights: Sequence[Policy],
    pairings: Sequence[Pairing],
    seed: int = 0,
    points_to_win: int = POINTS_TO_WIN,
) -> List[float]:
    with threadpool_limits(limits=MATH_THREADS_PER_PROCESS):
        return _play_matches_single_threaded(policy_weights, pairings, seed, points_to_win)


def _play_matches_single_threaded(
    policy_weights: Sequence[Policy],
    pairings: Sequence[Pairing],
    seed: int,
    points_to_win: int,
) -> List[float]:
    policy_index_by_slot = [policy_index for pairing in pairings for policy_index in pairing]
    policies = PolicyGroups(policy_weights, policy_index_by_slot)
    environment = PufferSlimeVolley(num_envs=len(pairings), num_agents=AGENTS_PER_MATCH)
    observations, _ = environment.reset(seed=seed)

    buttons = np.zeros((len(policy_index_by_slot), BUTTON_COUNT), dtype=int)
    left_points_won = np.zeros(len(pairings), dtype=int)
    right_points_won = np.zeros(len(pairings), dtype=int)
    left_returns = np.zeros(len(pairings), dtype=int)
    right_returns = np.zeros(len(pairings), dtype=int)
    last_ball_side = np.full(len(pairings), NO_SIDE)
    match_finished = np.zeros(len(pairings), dtype=bool)
    while not match_finished.all():
        policies.fill_buttons(observations, buttons)
        observations, rewards, terminals, _, _ = environment.step(buttons)
        still_playing = ~match_finished
        left_rewards = rewards[LEFT_SLOT::AGENTS_PER_MATCH]
        point_scored = left_rewards != 0
        left_points_won += still_playing & (left_rewards == LEFT_WON_POINT_REWARD)
        right_points_won += still_playing & (left_rewards == RIGHT_WON_POINT_REWARD)

        ball_side = _ball_side_seen_from_left(observations)
        ball_changed_side = (ball_side != NO_SIDE) & (last_ball_side != NO_SIDE) & (ball_side != last_ball_side)
        left_returns += still_playing & ball_changed_side & (last_ball_side > 0)
        right_returns += still_playing & ball_changed_side & (last_ball_side < 0)
        last_ball_side = np.where(ball_side != NO_SIDE, ball_side, last_ball_side)
        last_ball_side[point_scored] = NO_SIDE
        someone_reached_target = np.maximum(left_points_won, right_points_won) >= points_to_win
        episode_ended = terminals[LEFT_SLOT::AGENTS_PER_MATCH].astype(bool)
        newly_finished = still_playing & (someone_reached_target | episode_ended)
        if newly_finished.any():
            match_finished |= newly_finished
            buttons[np.repeat(match_finished, AGENTS_PER_MATCH)] = FORWARD_BACKWARD_JUMP_BY_ACTION_INDEX[NOTHING_ACTION_INDEX]
            policies.set_active_slots(_slots_of_unfinished_matches(match_finished))
    environment.close()
    left_reward = left_points_won + RETURN_OVER_NET_REWARD * left_returns
    right_reward = right_points_won + RETURN_OVER_NET_REWARD * right_returns
    return [_left_share_of_reward(left, right) for left, right in zip(left_reward, right_reward)]
