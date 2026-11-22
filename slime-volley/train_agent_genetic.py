import argparse
import copy
import csv
import itertools
import math
import multiprocessing as mp
import os
import random
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm

from .agent_genetic import INITIAL_RATING, GeneticAgent, reserve_agent_ids_through
from .puffer_arena import DRAW, KINEMATIC_BOT, POINTS_TO_WIN, WIN, play_matches

CHAMPION_PATH = Path(__file__).resolve().parent / "champions" / "agent_genetic_gen_1000.pth"
TOP_AGENTS_DIRECTORY = Path(__file__).resolve().parent / "champions" / "top_agents"
LADDER_DIRECTORY = Path(__file__).resolve().parent / "champions" / "ladder"
LADDER_RESULTS_FILENAME = "ladder_elo.csv"
TOP_AGENT_COUNT = 5
TOP_AGENT_FILENAME_PATTERN = re.compile(r"rank_(?P<rank>\d+)_gen_(?P<generation>\d+)_rating_(?P<rating>\d+)\.pth")
TRAINING_STATE_PATH = Path(__file__).resolve().parent / "champions" / "training_state.pth"
NUM_WORKERS = os.cpu_count() or 1
MATCH_TASKS_PER_WORKER = 3
MATCH_SEED_BITS = 31

BOOTSTRAP_ANCHOR_COUNT = 5
POOL_MEMBERS_PER_GENERATION = 5
POOL_GENERATION_WINDOW = 5
LADDER_INTERVAL = 10
LADDER_MAX_RUNGS = 10

LADDER_ROUND_ROBIN_GAMES_PER_PAIR = 20
LADDER_ROUND_ROBIN_MAX_PLAYERS = 25
RANDOM_ANCHOR_GENERATION = 0
RANDOM_ANCHOR_SEED = 0
BOT_AGENT_ID = -1
BOT_GENERATION = -1
BOT_CALIBRATION_GAMES_PER_ANCHOR = 40
SCRIPTED_LADDER_PLAYER_COUNT = 2
BRADLEY_TERRY_PRIOR_DRAWS = 1
BRADLEY_TERRY_MAX_ITERATIONS = 1000
BRADLEY_TERRY_TOLERANCE = 1e-9
ELO_SCALE = 400.0

RATING_SEARCH_FLOOR = 0.0
RATING_SEARCH_CEILING = 4000.0
RATING_SEARCH_ITERATIONS = 60


@dataclass(frozen=True)
class FrozenOpponent:
    agent_id: int
    generation: int
    weights: Union[Dict[str, torch.Tensor], str]
    rating: float


@dataclass
class ScriptedPlayer:
    agent_id: int
    policy: str
    rating: float = INITIAL_RATING

    def weights(self) -> str:
        return self.policy


@dataclass(frozen=True)
class OpponentRecord:
    opponent_id: int
    opponent_rating: float
    results: Tuple[float, ...]

    @property
    def average_score(self) -> float:
        return sum(self.results) / len(self.results)


def play_match(
    agent: GeneticAgent,
    opponent: GeneticAgent,
    games: int = 2,
    points_to_win: int = POINTS_TO_WIN,
) -> Tuple[float, ...]:
    agent_plays_left_by_game = [_agent_plays_left(game_number) for game_number in range(games)]
    left_results = play_matches(
        [agent.weights(), opponent.weights()],
        [(0, 1) if agent_plays_left else (1, 0) for agent_plays_left in agent_plays_left_by_game],
        seed=random.getrandbits(MATCH_SEED_BITS),
        points_to_win=points_to_win,
    )
    return tuple(
        _result_for_agent(left_result, agent_plays_left)
        for left_result, agent_plays_left in zip(left_results, agent_plays_left_by_game)
    )


def _agent_plays_left(game_number: int) -> bool:
    return game_number % 2 == 0


def _result_for_agent(left_result: float, agent_plays_left: bool) -> float:
    return left_result if agent_plays_left else WIN - left_result


def _play_matches_worker(match_task) -> List[float]:
    policy_weights, pairings, seed, points_to_win = match_task
    torch.set_num_threads(1)
    return play_matches(policy_weights, pairings, seed, points_to_win)


def _build_match_task(weights_by_player_key: Dict, pairing_batch: List[Tuple], points_to_win: int) -> Tuple:
    batch_player_keys = list(dict.fromkeys(player_key for pairing in pairing_batch for player_key in pairing))
    policy_index_by_player_key = {player_key: index for index, player_key in enumerate(batch_player_keys)}
    return (
        [weights_by_player_key[player_key] for player_key in batch_player_keys],
        [
            (policy_index_by_player_key[left_key], policy_index_by_player_key[right_key])
            for left_key, right_key in pairing_batch
        ],
        random.getrandbits(MATCH_SEED_BITS),
        points_to_win,
    )


def _play_pairings_in_parallel(
    worker_pool,
    weights_by_player_key: Dict,
    pairings: List[Tuple],
    points_to_win: int,
    progress_label: str,
) -> List[float]:
    match_tasks = [
        _build_match_task(weights_by_player_key, pairing_batch, points_to_win)
        for pairing_batch in _split_into_batches(pairings, NUM_WORKERS * MATCH_TASKS_PER_WORKER)
    ]
    left_results: List[float] = []
    for batch_left_results in tqdm(
        worker_pool.imap(_play_matches_worker, match_tasks),
        total=len(match_tasks),
        unit="batch",
        desc=progress_label,
        leave=False,
    ):
        left_results.extend(batch_left_results)
    return left_results


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


def _split_into_batches(items: List, batch_count: int) -> List[List]:
    batch_size = max(1, math.ceil(len(items) / batch_count))
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]


def _rate_against_opponents(
    worker_pool,
    agents: List[GeneticAgent],
    opponents: List[FrozenOpponent],
    games_per_opponent: int,
    points_to_win: int,
    progress_label: str,
) -> Dict[int, List[OpponentRecord]]:
    weights_by_agent_id = {opponent.agent_id: opponent.weights for opponent in opponents}
    weights_by_agent_id.update({agent.agent_id: agent.weights() for agent in agents})
    scheduled_games = [
        (agent.agent_id, opponent.agent_id, _agent_plays_left(game_number))
        for agent in agents
        for opponent in opponents
        if opponent.agent_id != agent.agent_id
        for game_number in range(games_per_opponent)
    ]
    pairings = [
        (agent_id, opponent_id) if agent_plays_left else (opponent_id, agent_id)
        for agent_id, opponent_id, agent_plays_left in scheduled_games
    ]
    left_results = _play_pairings_in_parallel(
        worker_pool, weights_by_agent_id, pairings, points_to_win, progress_label
    )

    results_by_matchup: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for (agent_id, opponent_id, agent_plays_left), left_result in zip(scheduled_games, left_results):
        results_by_matchup[(agent_id, opponent_id)].append(_result_for_agent(left_result, agent_plays_left))

    records_by_agent_id: Dict[int, List[OpponentRecord]] = {}
    for agent in agents:
        records = [
            OpponentRecord(
                opponent_id=opponent.agent_id,
                opponent_rating=opponent.rating,
                results=tuple(results_by_matchup[(agent.agent_id, opponent.agent_id)]),
            )
            for opponent in opponents
            if opponent.agent_id != agent.agent_id
        ]
        records_by_agent_id[agent.agent_id] = records
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


def calibrate_bot_rating(worker_pool, games_per_anchor: int = BOT_CALIBRATION_GAMES_PER_ANCHOR, points_to_win: int = POINTS_TO_WIN) -> float:
    bot_player = ScriptedPlayer(agent_id=BOT_AGENT_ID, policy=KINEMATIC_BOT)
    _rate_against_opponents(
        worker_pool, [bot_player], _bootstrap_anchors(), games_per_anchor, points_to_win, progress_label="bot calibration"
    )
    return bot_player.rating


def _bot_opponent(bot_rating: float) -> FrozenOpponent:
    return FrozenOpponent(agent_id=BOT_AGENT_ID, generation=BOT_GENERATION, weights=KINEMATIC_BOT, rating=bot_rating)


def _bot_report(champion_records: List[OpponentRecord]) -> Optional[str]:
    bot_records = [record for record in champion_records if record.opponent_id == BOT_AGENT_ID]
    if not bot_records:
        return None
    bot_record = bot_records[0]
    return f"kinematic bot (rating {bot_record.opponent_rating:.0f}): scored {bot_record.average_score:.0%}"


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
            f"scored {record.average_score:.0%}, rating predicts {expected_win_rate:.0%}"
        )
    return " | ".join(rung_summaries)


def _ladder_checkpoint_path(directory, generation: int) -> Path:
    return Path(directory) / f"gen_{generation:05d}.pth"


def _clear_ladder_checkpoints(directory) -> None:
    for stale_checkpoint in Path(directory).glob("gen_*.pth"):
        stale_checkpoint.unlink()


def _load_ladder_checkpoints(directory) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    return [
        (int(checkpoint.stem.split("_")[1]), torch.load(checkpoint, map_location="cpu"))
        for checkpoint in sorted(Path(directory).glob("gen_*.pth"))
    ]


def _seeded_random_agent_weights() -> Dict[str, torch.Tensor]:
    with torch.random.fork_rng():
        torch.manual_seed(RANDOM_ANCHOR_SEED)
        return GeneticAgent().weights()


def _evenly_spaced(items: List, max_count: int) -> List:
    if len(items) <= max_count:
        return items
    chosen_indices = sorted({round(slot * (len(items) - 1) / (max_count - 1)) for slot in range(max_count)})
    return [items[index] for index in chosen_indices]


def fit_elo_ratings(
    player_count: int,
    pair_results: Dict[Tuple[int, int], Sequence[float]],
    anchor_index: int = 0,
) -> List[float]:
    points_against = [[0.0] * player_count for _ in range(player_count)]
    games_against = [[0.0] * player_count for _ in range(player_count)]
    for (first_index, second_index), results in pair_results.items():
        games_played = len(results) + BRADLEY_TERRY_PRIOR_DRAWS
        first_points = sum(results) + DRAW * BRADLEY_TERRY_PRIOR_DRAWS
        points_against[first_index][second_index] += first_points
        points_against[second_index][first_index] += games_played - first_points
        games_against[first_index][second_index] += games_played
        games_against[second_index][first_index] += games_played

    strengths = [1.0] * player_count
    for _ in range(BRADLEY_TERRY_MAX_ITERATIONS):
        updated_strengths = []
        for player in range(player_count):
            expected_games_weight = sum(
                games_against[player][other] / (strengths[player] + strengths[other])
                for other in range(player_count)
                if games_against[player][other]
            )
            updated_strengths.append(sum(points_against[player]) / expected_games_weight)
        mean_log_strength = sum(math.log(strength) for strength in updated_strengths) / player_count
        updated_strengths = [strength / math.exp(mean_log_strength) for strength in updated_strengths]
        largest_change = max(
            abs(math.log(updated / previous)) for updated, previous in zip(updated_strengths, strengths)
        )
        strengths = updated_strengths
        if largest_change < BRADLEY_TERRY_TOLERANCE:
            break

    anchor_strength = strengths[anchor_index]
    return [INITIAL_RATING + ELO_SCALE * math.log10(strength / anchor_strength) for strength in strengths]


def _write_ladder_results(directory, ladder_ratings: List[Tuple[int, float, float]]) -> Path:
    results_path = Path(directory) / LADDER_RESULTS_FILENAME
    with results_path.open("w", newline="") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["generation", "elo", "round_robin_score"])
        for generation, elo, score in ladder_ratings:
            writer.writerow([generation, f"{elo:.1f}", f"{score:.3f}"])
    return results_path


def _round_robin_scores(player_count: int, pair_results: Dict[Tuple[int, int], Sequence[float]]) -> List[float]:
    points = [0.0] * player_count
    games = [0] * player_count
    for (first_index, second_index), results in pair_results.items():
        points[first_index] += sum(results)
        points[second_index] += len(results) - sum(results)
        games[first_index] += len(results)
        games[second_index] += len(results)
    return [player_points / max(1, player_games) for player_points, player_games in zip(points, games)]


def rate_ladder(
    worker_pool,
    ladder_directory=LADDER_DIRECTORY,
    games_per_pair: int = LADDER_ROUND_ROBIN_GAMES_PER_PAIR,
    max_players: int = LADDER_ROUND_ROBIN_MAX_PLAYERS,
    points_to_win: int = POINTS_TO_WIN,
) -> List[Tuple[int, float, float]]:
    saved_rungs = _evenly_spaced(_load_ladder_checkpoints(ladder_directory), max_players - SCRIPTED_LADDER_PLAYER_COUNT)
    if not saved_rungs:
        return []
    players = [(RANDOM_ANCHOR_GENERATION, _seeded_random_agent_weights()), (BOT_GENERATION, KINEMATIC_BOT)] + saved_rungs
    weights_by_player_index = {player_index: weights for player_index, (_, weights) in enumerate(players)}
    scheduled_games = [
        (first_index, second_index, _agent_plays_left(game_number))
        for first_index, second_index in itertools.combinations(range(len(players)), 2)
        for game_number in range(games_per_pair)
    ]
    pairings = [
        (first_index, second_index) if first_plays_left else (second_index, first_index)
        for first_index, second_index, first_plays_left in scheduled_games
    ]
    left_results = _play_pairings_in_parallel(
        worker_pool, weights_by_player_index, pairings, points_to_win, progress_label="ladder round robin"
    )

    pair_results: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for (first_index, second_index, first_plays_left), left_result in zip(scheduled_games, left_results):
        pair_results[(first_index, second_index)].append(_result_for_agent(left_result, first_plays_left))

    elo_ratings = fit_elo_ratings(len(players), pair_results, anchor_index=0)
    scores = _round_robin_scores(len(players), pair_results)
    ladder_ratings = [
        (generation, elo, score) for (generation, _), elo, score in zip(players, elo_ratings, scores)
    ]

    results_path = _write_ladder_results(ladder_directory, ladder_ratings)
    print(f"Ladder round robin ({games_per_pair} games per pair, random agent anchored at {INITIAL_RATING:.0f}):")
    for generation, elo, score in ladder_ratings:
        label = {RANDOM_ANCHOR_GENERATION: "random", BOT_GENERATION: "bot"}.get(generation, f"gen {generation}")
        print(f"  {label:>10}: Elo {elo:7.1f}   score {score:.0%}")
    print(f"Saved to {results_path}")
    return ladder_ratings


def rate_saved_ladder(
    ladder_directory=LADDER_DIRECTORY,
    games_per_pair: int = LADDER_ROUND_ROBIN_GAMES_PER_PAIR,
    max_players: int = LADDER_ROUND_ROBIN_MAX_PLAYERS,
) -> List[Tuple[int, float, float]]:
    with mp.get_context("spawn").Pool(processes=NUM_WORKERS) as worker_pool:
        return rate_ladder(worker_pool, ladder_directory, games_per_pair, max_players)


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
    bot_rating: Optional[float] = None


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
        "bot_rating": progress.bot_rating,
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
        bot_rating=serialized_state.get("bot_rating"),
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
    survival_rate: float = 0.1,
    mutation_std: float = 0.02,
    games_per_opponent: int = 2,
    finalist_count: int = 20,
    finalist_games_per_opponent: int = 4,
    points_to_win: int = POINTS_TO_WIN,
    champion_path=CHAMPION_PATH,
    top_agents_directory=TOP_AGENTS_DIRECTORY,
    training_state_path=TRAINING_STATE_PATH,
    resume: bool = False,
    seed_agents_directory=None,
    ladder_directory=LADDER_DIRECTORY,
    ladder_games_per_pair: int = LADDER_ROUND_ROBIN_GAMES_PER_PAIR,
) -> GeneticAgent:
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 < survival_rate <= 1.0:
        raise ValueError("survival_rate must be greater than 0 and at most 1")
    if games_per_opponent < 1 or finalist_games_per_opponent < 1:
        raise ValueError("games per opponent must be at least 1")
    if resume and seed_agents_directory is not None:
        raise ValueError("choose either resume or seed_agents_directory, not both")
    if finalist_count < 1:
        raise ValueError("finalist_count must be at least 1")

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
    Path(ladder_directory).mkdir(parents=True, exist_ok=True)
    if not resume:
        _clear_ladder_checkpoints(ladder_directory)

    multiprocessing_context = mp.get_context("spawn")
    with multiprocessing_context.Pool(processes=NUM_WORKERS) as worker_pool:
        if progress.bot_rating is None:
            progress.bot_rating = calibrate_bot_rating(worker_pool, points_to_win=points_to_win)
            print(f"Kinematic bot calibrated at rating {progress.bot_rating:.0f} against random agents at {INITIAL_RATING:.0f}")
        bot_opponent = _bot_opponent(progress.bot_rating)

        for generation in tqdm(
            range(progress.completed_generation + 1, generations + 1),
            unit="generation",
        ):
            opponents = (_build_opponent_pool(recent_generations, ladder) or _bootstrap_anchors()) + [bot_opponent]

            records_by_agent_id = _rate_against_opponents(
                worker_pool,
                population,
                opponents,
                games_per_opponent,
                points_to_win,
                progress_label="population",
            )
            population.sort(key=lambda agent: agent.rating, reverse=True)

            finalists = population[:finalist_count]
            finalist_records = _rate_against_opponents(
                worker_pool,
                finalists,
                opponents,
                finalist_games_per_opponent,
                points_to_win,
                progress_label="finalists",
            )
            records_by_agent_id.update(finalist_records)
            finalists.sort(key=lambda agent: agent.rating, reverse=True)
            population = finalists + population[finalist_count:]

            pool_entrants = _select_pool_entrants(finalists, opponents)

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
            bot_summary = _bot_report(records_by_agent_id[champion.agent_id])
            if bot_summary:
                print(f"  Benchmark: {bot_summary}")

            recent_generations.append([_freeze(agent, generation) for agent in pool_entrants])
            is_final_generation = generation == generations
            if generation % LADDER_INTERVAL == 0:
                ladder = _add_ladder_rung(ladder, _freeze(champion, generation))
            if generation % LADDER_INTERVAL == 0 or is_final_generation:
                champion.save_model(_ladder_checkpoint_path(ladder_directory, generation))

            population = _next_generation(population, population_size, survival_rate, mutation_std)
            save_training_state(
                TrainingProgress(generation, population, recent_generations, ladder, progress.bot_rating),
                training_state_path,
            )

        rate_ladder(worker_pool, ladder_directory, ladder_games_per_pair, points_to_win=points_to_win)

    return champion


def _parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve slime volleyball agents.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=100)
    starting_point = parser.add_mutually_exclusive_group()
    starting_point.add_argument("--resume", action="store_true")
    starting_point.add_argument("--seed-from", type=Path, default=None)
    starting_point.add_argument("--rate-ladder", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    command_line = _parse_command_line()
    if command_line.rate_ladder:
        rate_saved_ladder()
    else:
        train(
            generations=command_line.generations,
            population_size=command_line.population_size,
            resume=command_line.resume,
            seed_agents_directory=command_line.seed_from,
        )
