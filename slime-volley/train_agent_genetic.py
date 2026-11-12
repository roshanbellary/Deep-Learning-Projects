"""Self-play evaluation helpers for the genetic-agent population."""

import copy
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
import random
import torch
from tqdm import tqdm
from .agent_genetic import BaseAgentModel, GeneticAgent, GeneticAgentWithGradSensitivity
from .game import Action, GameEvent, Player, SlimeVolleyball


CHAMPION_PATH = Path(__file__).resolve().parent / "champions" / "agent_genetic_gen_1000.pth"
NUM_WORKERS = 8


# Scores are deliberately event-based. The game emits only events; this table
# makes the genetic fitness signal explicit and easy to tune.
EVENT_SCORES = {
    GameEvent.PLAYER_ONE_HIT: 2.0,
    GameEvent.PLAYER_TWO_HIT: 2.0,
    GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE: 4.0,
    GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE: 4.0,
    GameEvent.PLAYER_ONE_WON_ROUND: 100.0,
    GameEvent.PLAYER_TWO_WON_ROUND: 100.0,
}


def _choose_action(agent: GeneticAgentWithGradSensitivity, state: List[float]) -> Action:
    """Run one agent policy and convert its output to a game action."""

    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    with torch.inference_mode():
        action_index = agent.move(state_tensor)
    if hasattr(action_index, "item"):
        action_index = action_index.item()
    return list(Action)[int(action_index)]


HIT_REWARD = 2.0


def _score_events(events, hit_counts: Dict[Player, int]) -> Tuple[float, float]:
    """Translate events into points and update per-side hit counts."""

    player_one_points = 0.0
    player_two_points = 0.0

    # A crossing starts a new rally segment. Clear both counters before
    # processing any hit that may occur in the same simulation step.
    if (
        GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE in events
        or GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE in events
    ):
        hit_counts[Player.PLAYER_ONE] = 0
        hit_counts[Player.PLAYER_TWO] = 0

    for event in events:
        if event is GameEvent.PLAYER_ONE_HIT:
            count = hit_counts[Player.PLAYER_ONE]
            player_one_points += (HIT_REWARD - (float(HIT_REWARD/4) * count))
            hit_counts[Player.PLAYER_ONE] += 1
        elif event is GameEvent.PLAYER_TWO_HIT:
            count = hit_counts[Player.PLAYER_TWO]
            player_two_points += HIT_REWARD - (float(HIT_REWARD/4) * count)
            hit_counts[Player.PLAYER_TWO] += 1
        elif event is GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE:
            player_one_points += EVENT_SCORES[event]
        elif event is GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE:
            player_two_points += EVENT_SCORES[event]
        elif event is GameEvent.PLAYER_ONE_WON_ROUND:
            player_one_points += EVENT_SCORES[event]
        elif event is GameEvent.PLAYER_TWO_WON_ROUND:
            player_two_points += EVENT_SCORES[event]

    # Return points by court position. The caller maps these to agent1/agent2.
    return player_one_points, player_two_points


def _update_elo(agent1: GeneticAgentWithGradSensitivity, agent2: GeneticAgentWithGradSensitivity, winner: int, k_factor: float = 32.0):
    """Update the optional Elo values after one scored round.

    ``winner`` is 1 for agent1, 2 for agent2, and 0 for a draw.
    """

    expected_agent1 = 1 / (1 + 10 ** ((agent2.elo_score - agent1.elo_score) / 400))
    actual_agent1 = 1.0 if winner == 1 else 0.0 if winner == 2 else 0.5
    delta = k_factor * (actual_agent1 - expected_agent1)
    agent1.elo_score += delta
    agent2.elo_score -= delta


def _simulate_games(
    agent1: GeneticAgentWithGradSensitivity,
    agent2: GeneticAgentWithGradSensitivity,
    games: int = 4,
    max_steps_per_round: int = 10_000,
    update_elo: bool = True,
) -> Dict[str, object]:
    """Play ``games`` total rounds, alternating which side each agent plays."""

    agent1.model.eval()
    agent2.model.eval()
    game = SlimeVolleyball(render=False)
    rounds: List[Dict[str, object]] = []
    agent1_total_score = 0.0
    agent2_total_score = 0.0
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    # Alternate court assignments so the default two games evaluate each
    # agent once on each side.
    for game_number in range(games):
        agent1_player = (
            Player.PLAYER_ONE
            if game_number % 2 == 0
            else Player.PLAYER_TWO
        )
        states = game.reset()
        agent1_score = 0.0
        agent2_score = 0.0
        hit_counts = {
            Player.PLAYER_ONE: 0,
            Player.PLAYER_TWO: 0,
        }
        result = None

        for step_number in range(1, max_steps_per_round + 1):
            # get_states() mirrors player two's state. Selecting by court
            # assignment is the only side-specific logic the agents need.
            player_one_state = states[Player.PLAYER_ONE.value]
            player_two_state = states[Player.PLAYER_TWO.value]
            action_one = _choose_action(
                agent1 if agent1_player is Player.PLAYER_ONE else agent2,
                player_one_state,
            )
            action_two = _choose_action(
                agent1 if agent1_player is Player.PLAYER_TWO else agent2,
                player_two_state,
            )

            result = game.step(action_one, action_two)
            points_one, points_two = _score_events(result.events, hit_counts)
            if agent1_player is Player.PLAYER_ONE:
                agent1_score += points_one
                agent2_score += points_two
            else:
                agent1_score += points_two
                agent2_score += points_one
            states = result.states

            if result.done:
                break

        if result is None:
            raise RuntimeError("The game produced no step result")

        if agent1_score > agent2_score:
            winner = 1
            agent1_wins += 1
        elif agent2_score > agent1_score:
            winner = 2
            agent2_wins += 1
        else:
            winner = 0
            draws += 1
        if update_elo:
            _update_elo(agent1, agent2, winner)

        agent1_total_score += agent1_score
        agent2_total_score += agent2_score
        rounds.append(
            {
                "agent1_score": agent1_score,
                "agent2_score": agent2_score,
                "winner": winner,
                "agent1_player": agent1_player.value,
                "steps": step_number,
                "completed": result.done,
            }
        )
    return {
        "agent1_score": agent1_total_score,
        "agent2_score": agent2_total_score,
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "games_played": len(rounds),
        "rounds": rounds,
    }


def _simulate_matchup_worker(task):
    """Run one matchup in a child process and return its parent indices."""

    first_index, second_index, agent1, agent2, max_steps_per_round = task
    result = _simulate_games(
        agent1,
        agent2,
        max_steps_per_round=max_steps_per_round,
        update_elo=False,
    )
    return first_index, second_index, result


def simulate_games(
    agent1: GeneticAgentWithGradSensitivity,
    agent2: GeneticAgentWithGradSensitivity,
    games: int = 2,
    max_steps_per_round: int = 10_000,
) -> Dict[str, object]:
    """Public wrapper for two-game agent evaluation."""

    return _simulate_games(agent1, agent2, games, max_steps_per_round)


def train(
    generations: int = 100,
    population_size: int = 100,
    opponent_sample_ratio: float = 0.2,
    max_steps_per_round: int = 10_000,
    survival_rate: float = 0.1,
    champion_path=CHAMPION_PATH,
):
    """Run evolution and save the highest-Elo final champion.

    The default checkpoint is ``champions/agent_genetic_gen_1000.pth`` next
    to this module. ``champion_path`` can be overridden for a shorter test
    run or a different generation count.
    """

    if generations < 1:
        raise ValueError("generations must be at least 1")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 < opponent_sample_ratio <= 1.0:
        raise ValueError("opponent_sample_ratio must be greater than 0 and at most 1")
    if not 0.0 < survival_rate <= 1.0:
        raise ValueError("survival_rate must be greater than 0 and at most 1")

    population = [GeneticAgentWithGradSensitivity(BaseAgentModel()) for _ in range(population_size)]
    champions = []

    for generation in tqdm(range(1, generations + 1), unit="generation"):
        matchups = set()
        opponent_count = max(1, int(len(population) * opponent_sample_ratio))
        opponent_count = min(opponent_count, len(population) - 1)

        for index in range(len(population)):
            opponent_indices = random.sample(
                [candidate for candidate in range(len(population)) if candidate != index],
                opponent_count
            )
            for opponent_index in opponent_indices:
                matchups.add(tuple(sorted((index, opponent_index))))

        matchup_tasks = [
            (
                first_index,
                second_index,
                population[first_index],
                population[second_index],
                max_steps_per_round,
            )
            for first_index, second_index in matchups
        ]

        # Workers return results only. Elo is applied in the parent process in
        # matchup order, avoiding shared-state races.
        multiprocessing_context = mp.get_context("spawn")
        with multiprocessing_context.Pool(processes=NUM_WORKERS) as pool:
            matchup_results = pool.imap(_simulate_matchup_worker, matchup_tasks)
            for first_index, second_index, match_result in tqdm(
                matchup_results,
                total=len(matchup_tasks),
                unit="matchup",
            ):
                for round_result in match_result["rounds"]:
                    _update_elo(
                        population[first_index],
                        population[second_index],
                        round_result["winner"],
                    )

        population.sort(key=lambda agent: agent.elo_score, reverse=True)
        champion = population[0]
        champions.append(copy.deepcopy(champion))
        saved_path = champions[-1].save_model(champion_path)
        print(f"Saved champion model to {saved_path}")

        print(
            f"Generation {generation} concluded. "
            f"Champion Elo: {champion.elo_score:.2f}"
        )

        survivor_count = max(1, int(len(population) * survival_rate))
        survivors = population[:survivor_count]
        for survivor in survivors:
            survivor.elo_score = 1000

        population = list(survivors)
        while (len(population) < population_size):
            chosen_agent = random.choice(survivors)
            population.append(chosen_agent.mutate())
        
        for survivor in survivors:
            mutation_count = int(1/survival_rate - 1)
            population.extend([survivor.mutate() for _ in range(mutation_count)])
    best_agent = champions[-1]
    return best_agent


if __name__ == "__main__":
    train()
