import argparse
import copy
import math
import multiprocessing as mp
import os
import random
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .agent_genetic import INITIAL_RATING, GeneticAgent, reserve_agent_ids_through
from .game import Action, GameEvent, Player, SlimeVolleyball

CHAMPION_PATH = Path(__file__).resolve().parent / "champions" / "agent_genetic_gen_1000.pth"
TOP_AGENTS_DIRECTORY = Path(__file__).resolve().parent / "champions" / "top_agents"
TOP_AGENT_COUNT = 5
TOP_AGENT_FILENAME_PATTERN = re.compile(r"rank_(?P<rank>\d+)_gen_(?P<generation>\d+)_rating_(?P<rating>\d+)\.pth")
TRAINING_STATE_PATH = Path(__file__).resolve().parent / "champions" / "training_state.pth"
NUM_WORKERS = os.cpu_count() or 1
EVALUATION_TASKS_PER_WORKER = 3

FIRST_HIT_REWARD = 2.0
HIT_REWARD_DECAY = 0.5
CROSSING_REWARD = 4.0
ROUND_WIN_REWARD = 100.0

SERVE_SIDEWAYS_JITTER = 0.5

BOOTSTRAP_ANCHOR_COUNT = 5
POOL_MEMBERS_PER_GENERATION = 5
POOL_GENERATION_WINDOW = 5
LADDER_INTERVAL = 10
LADDER_MAX_RUNGS = 10

RATING_SEARCH_FLOOR = 0.0
RATING_SEARCH_CEILING = 4000.0
RATING_SEARCH_ITERATIONS = 60

WIN = 1.0
DRAW = 0.5
LOSS = 0.0

ACTIONS_BY_INDEX = list(Action)
CROSSING_EVENTS = {
    GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE,
    GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE,
}


@dataclass(frozen=True)
class FrozenOpponent:
    agent_id: int
    generation: int
    weights: Dict[str, torch.Tensor]
    rating: float


@dataclass(frozen=True)
class OpponentRecord:
    opponent_id: int
    opponent_rating: float
    results: Tuple[float, ...]

    @property
    def win_rate(self) -> float:
        return sum(self.results) / len(self.results)


def _choose_action(agent: GeneticAgent, raw_observation: List[float]) -> Action:
    observation_tensor = torch.as_tensor(raw_observation, dtype=torch.float32)
    with torch.inference_mode():
        action_index = agent.move(observation_tensor)
    return ACTIONS_BY_INDEX[int(action_index)]


def _decayed_hit_reward(consecutive_hit_count: int) -> float:
    return FIRST_HIT_REWARD * (HIT_REWARD_DECAY ** consecutive_hit_count)


def _score_events(
    events,
    consecutive_hits: Dict[Player, int],
    use_shaped_rewards: bool,
) -> Tuple[float, float]:
    player_one_points = 0.0
    player_two_points = 0.0

    if events & CROSSING_EVENTS:
        consecutive_hits[Player.PLAYER_ONE] = 0
        consecutive_hits[Player.PLAYER_TWO] = 0

    for event in events:
        if event is GameEvent.PLAYER_ONE_WON_ROUND:
            player_one_points += ROUND_WIN_REWARD
        elif event is GameEvent.PLAYER_TWO_WON_ROUND:
            player_two_points += ROUND_WIN_REWARD
        elif not use_shaped_rewards:
            continue
        elif event is GameEvent.PLAYER_ONE_HIT:
            player_one_points += _decayed_hit_reward(consecutive_hits[Player.PLAYER_ONE])
            consecutive_hits[Player.PLAYER_ONE] += 1
        elif event is GameEvent.PLAYER_TWO_HIT:
            player_two_points += _decayed_hit_reward(consecutive_hits[Player.PLAYER_TWO])
            consecutive_hits[Player.PLAYER_TWO] += 1
        elif event is GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE:
            player_one_points += CROSSING_REWARD
        elif event is GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE:
            player_two_points += CROSSING_REWARD

    return player_one_points, player_two_points


def _play_one_round(
    game: SlimeVolleyball,
    agent: GeneticAgent,
    opponent: GeneticAgent,
    agent_side: Player,
    max_steps_per_round: int,
    use_shaped_rewards: bool,
) -> float:
    player_one_agent = agent if agent_side is Player.PLAYER_ONE else opponent
    player_two_agent = opponent if agent_side is Player.PLAYER_ONE else agent
    observations = game.reset()
    consecutive_hits = {Player.PLAYER_ONE: 0, Player.PLAYER_TWO: 0}
    player_one_total = 0.0
    player_two_total = 0.0

    for _ in range(max_steps_per_round):
        step_result = game.step(
            _choose_action(player_one_agent, observations[Player.PLAYER_ONE.value]),
            _choose_action(player_two_agent, observations[Player.PLAYER_TWO.value]),
        )
        player_one_points, player_two_points = _score_events(
            step_result.events, consecutive_hits, use_shaped_rewards
        )
        player_one_total += player_one_points
        player_two_total += player_two_points
        observations = step_result.states
        if step_result.done:
            break

    agent_total = player_one_total if agent_side is Player.PLAYER_ONE else player_two_total
    opponent_total = player_two_total if agent_side is Player.PLAYER_ONE else player_one_total
    if agent_total > opponent_total:
        return WIN
    if agent_total < opponent_total:
        return LOSS
    return DRAW


def play_match(
    agent: GeneticAgent,
    opponent: GeneticAgent,
    games: int = 2,
    max_steps_per_round: int = 10_000,
    use_shaped_rewards: bool = True,
    serve_sideways_jitter: float = SERVE_SIDEWAYS_JITTER,
) -> Tuple[float, ...]:
    agent.model.eval()
    opponent.model.eval()
    game = SlimeVolleyball(render=False, serve_sideways_jitter=serve_sideways_jitter)
    return tuple(
        _play_one_round(
            game,
            agent,
            opponent,
            Player.PLAYER_ONE if game_number % 2 == 0 else Player.PLAYER_TWO,
            max_steps_per_round,
            use_shaped_rewards,
        )
        for game_number in range(games)
    )


def _expected_score(rating: float, opponent_rating: float) -> float:
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def performance_rating(opponent_ratings: Sequence[float], results: Sequence[float]) -> float:
    prior_opponent_rating = sum(opponent_ratings) / len(opponent_ratings)
    rated_opponents = list(opponent_ratings) + [prior_opponent_rating]
    total_score = sum(results) + DRAW

    rating_floor = RATING_SEARCH_FLOOR
    rating_ceiling = RATING_SEARCH_CEILING
    for _ in range(RATING_SEARCH_ITERATIONS):
        rating_guess = (rating_floor + rating_ceiling) / 2
        expected_total = sum(_expected_score(rating_guess, rating) for rating in rated_opponents)
        if expected_total < total_score:
            rating_floor = rating_guess
        else:
            rating_ceiling = rating_guess
    return (rating_floor + rating_ceiling) / 2


def _evaluate_agents_worker(evaluation_task):
    (
        agent_ids,
        agent_weights,
        opponents,
        games_per_opponent,
        max_steps_per_round,
        use_shaped_rewards,
    ) = evaluation_task
    torch.set_num_threads(1)
    opponent_agents = [
        (opponent, GeneticAgent.from_weights(opponent.weights)) for opponent in opponents
    ]
    evaluations = []
    for agent_id, weights in zip(agent_ids, agent_weights):
        agent = GeneticAgent.from_weights(weights)
        records = [
            OpponentRecord(
                opponent_id=opponent.agent_id,
                opponent_rating=opponent.rating,
                results=play_match(
                    agent,
                    opponent_agent,
                    games=games_per_opponent,
                    max_steps_per_round=max_steps_per_round,
                    use_shaped_rewards=use_shaped_rewards,
                ),
            )
            for opponent, opponent_agent in opponent_agents
            if opponent.agent_id != agent_id
        ]
        evaluations.append((agent_id, records))
    return evaluations


def _split_into_batches(items: List, batch_count: int) -> List[List]:
    batch_size = max(1, math.ceil(len(items) / batch_count))
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]


def _rate_against_opponents(
    worker_pool,
    agents: List[GeneticAgent],
    opponents: List[FrozenOpponent],
    games_per_opponent: int,
    max_steps_per_round: int,
    use_shaped_rewards: bool,
    progress_label: str,
) -> Dict[int, List[OpponentRecord]]:
    agent_batches = _split_into_batches(agents, NUM_WORKERS * EVALUATION_TASKS_PER_WORKER)
    evaluation_tasks = [
        (
            [agent.agent_id for agent in batch],
            [agent.weights() for agent in batch],
            opponents,
            games_per_opponent,
            max_steps_per_round,
            use_shaped_rewards,
        )
        for batch in agent_batches
    ]

    records_by_agent_id: Dict[int, List[OpponentRecord]] = {}
    for batch_evaluations in tqdm(
        worker_pool.imap_unordered(_evaluate_agents_worker, evaluation_tasks),
        total=len(evaluation_tasks),
        unit="batch",
        desc=progress_label,
        leave=False,
    ):
        for agent_id, records in batch_evaluations:
            records_by_agent_id[agent_id] = records

    for agent in agents:
        records = records_by_agent_id[agent.agent_id]
        agent.rating = performance_rating(
            [record.opponent_rating for record in records for _ in record.results],
            [result for record in records for result in record.results],
        )
    return records_by_agent_id


def _freeze(agent: GeneticAgent, generation: int) -> FrozenOpponent:
    return FrozenOpponent(
        agent_id=agent.agent_id,
        generation=generation,
        weights=copy.deepcopy(agent.weights()),
        rating=agent.rating,
    )


def _bootstrap_anchors() -> List[FrozenOpponent]:
    anchors = []
    for _ in range(BOOTSTRAP_ANCHOR_COUNT):
        random_agent = GeneticAgent()
        random_agent.rating = INITIAL_RATING
        anchors.append(_freeze(random_agent, generation=0))
    return anchors


def _build_opponent_pool(
    recent_generations: Deque[List[FrozenOpponent]],
    ladder: List[FrozenOpponent],
) -> List[FrozenOpponent]:
    opponents_by_id: Dict[int, FrozenOpponent] = {}
    for opponent in [member for generation in recent_generations for member in generation] + ladder:
        opponents_by_id.setdefault(opponent.agent_id, opponent)
    return list(opponents_by_id.values())


def _select_pool_entrants(
    ranked_population: List[GeneticAgent],
    current_pool: List[FrozenOpponent],
) -> List[GeneticAgent]:
    pooled_ids = {opponent.agent_id for opponent in current_pool}
    fresh_agents = [agent for agent in ranked_population if agent.agent_id not in pooled_ids]
    return fresh_agents[:POOL_MEMBERS_PER_GENERATION]


def _add_ladder_rung(ladder: List[FrozenOpponent], rung: FrozenOpponent) -> List[FrozenOpponent]:
    extended_ladder = ladder + [rung]
    if len(extended_ladder) > LADDER_MAX_RUNGS:
        return extended_ladder[::2]
    return extended_ladder


def _ladder_report(
    champion: GeneticAgent,
    champion_records: List[OpponentRecord],
    ladder: List[FrozenOpponent],
) -> Optional[str]:
    records_by_opponent_id = {record.opponent_id: record for record in champion_records}
    played_rungs = [rung for rung in ladder if rung.agent_id in records_by_opponent_id]
    if not played_rungs:
        return None

    labelled_rungs = [("oldest", played_rungs[0]), ("newest", played_rungs[-1])]
    if len(played_rungs) == 1:
        labelled_rungs = [("only", played_rungs[0])]

    rung_summaries = []
    for label, rung in labelled_rungs:
        record = records_by_opponent_id[rung.agent_id]
        expected_win_rate = _expected_score(champion.rating, rung.rating)
        rung_summaries.append(
            f"{label} rung (gen {rung.generation}, rating {rung.rating:.0f}): "
            f"won {record.win_rate:.0%}, rating predicts {expected_win_rate:.0%}"
        )
    return " | ".join(rung_summaries)


def save_top_agents(ranked_population: List[GeneticAgent], generation: int, directory=TOP_AGENTS_DIRECTORY) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale_checkpoint in directory.glob("rank_*.pth"):
        stale_checkpoint.unlink()
    return [
        agent.save_model(directory / f"rank_{rank}_gen_{generation}_rating_{agent.rating:.0f}.pth")
        for rank, agent in enumerate(ranked_population[:TOP_AGENT_COUNT], start=1)
    ]


@dataclass
class TrainingProgress:
    completed_generation: int
    population: List[GeneticAgent]
    recent_generations: Deque[List[FrozenOpponent]]
    ladder: List[FrozenOpponent]


def _fresh_progress(population_size: int) -> TrainingProgress:
    return TrainingProgress(
        completed_generation=0,
        population=[GeneticAgent() for _ in range(population_size)],
        recent_generations=deque(maxlen=POOL_GENERATION_WINDOW),
        ladder=[],
    )


def _restore_agent(agent_id: int, rating: float, weights: Dict[str, torch.Tensor]) -> GeneticAgent:
    agent = GeneticAgent.from_weights(weights)
    agent.agent_id = agent_id
    agent.rating = rating
    return agent


def save_training_state(progress: TrainingProgress, path=TRAINING_STATE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_state = {
        "completed_generation": progress.completed_generation,
        "population": [
            {"agent_id": agent.agent_id, "rating": agent.rating, "weights": agent.weights()}
            for agent in progress.population
        ],
        "recent_generations": [
            [asdict(opponent) for opponent in generation_members]
            for generation_members in progress.recent_generations
        ],
        "ladder": [asdict(rung) for rung in progress.ladder],
    }
    partially_written_path = path.with_suffix(".partial")
    torch.save(serialized_state, partially_written_path)
    os.replace(partially_written_path, path)
    return path


def load_training_state(path=TRAINING_STATE_PATH) -> TrainingProgress:
    serialized_state = torch.load(path, map_location="cpu")
    progress = TrainingProgress(
        completed_generation=serialized_state["completed_generation"],
        population=[_restore_agent(**agent_fields) for agent_fields in serialized_state["population"]],
        recent_generations=deque(
            (
                [FrozenOpponent(**opponent_fields) for opponent_fields in generation_members]
                for generation_members in serialized_state["recent_generations"]
            ),
            maxlen=POOL_GENERATION_WINDOW,
        ),
        ladder=[FrozenOpponent(**rung_fields) for rung_fields in serialized_state["ladder"]],
    )
    _reserve_ids_used_by(progress)
    return progress


def _reserve_ids_used_by(progress: TrainingProgress):
    used_agent_ids = [agent.agent_id for agent in progress.population]
    used_agent_ids += [opponent.agent_id for members in progress.recent_generations for opponent in members]
    used_agent_ids += [rung.agent_id for rung in progress.ladder]
    reserve_agent_ids_through(max(used_agent_ids))


def load_top_agents(directory=TOP_AGENTS_DIRECTORY) -> Tuple[List[GeneticAgent], int]:
    ranked_checkpoints = []
    for checkpoint_path in Path(directory).glob("rank_*.pth"):
        filename_match = TOP_AGENT_FILENAME_PATTERN.fullmatch(checkpoint_path.name)
        if filename_match:
            ranked_checkpoints.append((int(filename_match["rank"]), filename_match, checkpoint_path))
    if not ranked_checkpoints:
        raise FileNotFoundError(f"No rank_*_gen_*_rating_*.pth checkpoints in {directory}")
    ranked_checkpoints.sort(key=lambda checkpoint: checkpoint[0])

    top_agents = []
    for _, filename_match, checkpoint_path in ranked_checkpoints:
        agent = GeneticAgent.load_model(checkpoint_path)
        agent.rating = float(filename_match["rating"])
        top_agents.append(agent)
    saved_generation = max(int(filename_match["generation"]) for _, filename_match, _ in ranked_checkpoints)
    return top_agents, saved_generation


def seed_progress_from_top_agents(
    population_size: int,
    mutation_std: float,
    directory=TOP_AGENTS_DIRECTORY,
) -> TrainingProgress:
    seed_agents, saved_generation = load_top_agents(directory)
    mutated_children = [
        random.choice(seed_agents).mutate(mutation_std)
        for _ in range(population_size - len(seed_agents))
    ]
    return TrainingProgress(
        completed_generation=saved_generation,
        population=seed_agents + mutated_children,
        recent_generations=deque(
            [[_freeze(agent, saved_generation) for agent in seed_agents]],
            maxlen=POOL_GENERATION_WINDOW,
        ),
        ladder=[],
    )


def _next_generation(
    ranked_population: List[GeneticAgent],
    population_size: int,
    survival_rate: float,
    mutation_std: float,
) -> List[GeneticAgent]:
    survivor_count = max(1, int(population_size * survival_rate))
    survivors = ranked_population[:survivor_count]
    offspring = [
        random.choice(survivors).mutate(mutation_std)
        for _ in range(population_size - survivor_count)
    ]
    return survivors + offspring


def train(
    generations: int = 100,
    population_size: int = 100,
    max_steps_per_round: int = 10_000,
    survival_rate: float = 0.1,
    mutation_std: float = 0.02,
    games_per_opponent: int = 2,
    entry_games_per_opponent: int = 6,
    use_shaped_rewards: bool = True,
    champion_path=CHAMPION_PATH,
    top_agents_directory=TOP_AGENTS_DIRECTORY,
    training_state_path=TRAINING_STATE_PATH,
    resume: bool = False,
    seed_agents_directory=None,
) -> GeneticAgent:
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 < survival_rate <= 1.0:
        raise ValueError("survival_rate must be greater than 0 and at most 1")
    if games_per_opponent < 1 or entry_games_per_opponent < 1:
        raise ValueError("games per opponent must be at least 1")
    if resume and seed_agents_directory is not None:
        raise ValueError("choose either resume or seed_agents_directory, not both")

    if resume:
        progress = load_training_state(training_state_path)
    elif seed_agents_directory is not None:
        progress = seed_progress_from_top_agents(population_size, mutation_std, seed_agents_directory)
    else:
        progress = _fresh_progress(population_size)
    population = progress.population
    recent_generations = progress.recent_generations
    ladder = progress.ladder
    champion: Optional[GeneticAgent] = None

    multiprocessing_context = mp.get_context("spawn")
    with multiprocessing_context.Pool(processes=NUM_WORKERS) as worker_pool:
        for generation in tqdm(
            range(progress.completed_generation + 1, generations + 1),
            unit="generation",
        ):
            opponents = _build_opponent_pool(recent_generations, ladder) or _bootstrap_anchors()

            records_by_agent_id = _rate_against_opponents(
                worker_pool,
                population,
                opponents,
                games_per_opponent,
                max_steps_per_round,
                use_shaped_rewards,
                progress_label="population",
            )
            population.sort(key=lambda agent: agent.rating, reverse=True)

            pool_entrants = _select_pool_entrants(population, opponents)
            entrant_records = _rate_against_opponents(
                worker_pool,
                pool_entrants,
                opponents,
                entry_games_per_opponent,
                max_steps_per_round,
                use_shaped_rewards,
                progress_label="pool entrants",
            )
            records_by_agent_id.update(entrant_records)
            population.sort(key=lambda agent: agent.rating, reverse=True)

            champion = copy.deepcopy(population[0])
            saved_path = champion.save_model(champion_path)
            save_top_agents(population, generation, top_agents_directory)
            print(
                f"Generation {generation}: champion rating {champion.rating:.0f} "
                f"against {len(opponents)} opponents. Saved to {saved_path}"
            )
            ladder_summary = _ladder_report(champion, records_by_agent_id[champion.agent_id], ladder)
            if ladder_summary:
                print(f"  Ladder: {ladder_summary}")

            recent_generations.append([_freeze(agent, generation) for agent in pool_entrants])
            if generation % LADDER_INTERVAL == 0:
                ladder = _add_ladder_rung(ladder, _freeze(champion, generation))

            population = _next_generation(population, population_size, survival_rate, mutation_std)
            save_training_state(
                TrainingProgress(generation, population, recent_generations, ladder),
                training_state_path,
            )

    return champion


def _parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve slime volleyball agents.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=100)
    starting_point = parser.add_mutually_exclusive_group()
    starting_point.add_argument("--resume", action="store_true")
    starting_point.add_argument("--seed-from", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    command_line = _parse_command_line()
    train(
        generations=command_line.generations,
        population_size=command_line.population_size,
        resume=command_line.resume,
        seed_agents_directory=command_line.seed_from,
    )
