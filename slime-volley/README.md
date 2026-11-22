# Slime Volleyball AI

A reinforcement learning implementation of the classic Slime Volleyball game where AI agents learn to play using Deep Q-Networks (DQN).

## Overview

This project implements a two-player Slime Volleyball game with AI agents that learn to play through reinforcement learning. The game features physics-based gameplay with gravity, collisions, and realistic ball dynamics. Two AI agents are trained simultaneously to compete against each other, improving their strategies over time.

## Features

- **Physics-based gameplay**: Realistic ball movement, gravity, and collision detection
- **AI vs AI training**: Two independent DQN agents learning simultaneously
- **Visual gameplay**: Real-time rendering with pygame showing slimes with animated eyes that track the ball
- **Reward system**: Agents receive rewards for hitting the ball and scoring points
- **Customizable training**: Adjustable hyperparameters for learning rate, exploration, and network architecture

## Game Mechanics

- **Controls**: Move left/right, jump, or stay still (4 possible actions)
- **Objective**: Hit the ball over the net to score on the opponent's side
- **Physics**: Realistic ball bouncing, paddle collisions, and gravity effects
- **Proportions**: Sizes and speeds follow the `slimevolleygym` reference
  (48-unit-wide arena, slime radius 1.5, ball radius 0.5, net 1 wide and 3.5
  tall, gravity 29.4, jump speed 13.5, run speed 17.5, ball speed cap 22.5).
  They are defined in world units at the top of `game.py` and converted to
  pixels per frame at 60 FPS.
- **Slimes**: Half-disc bodies. The dome and the flat underside both collide
  with the ball, and the slime's horizontal and vertical velocity is transferred
  into the ball, so running into the ball pushes it.
- **Net**: A solid post. The ball bounces off its sides and its top.
- **Serve**: The ball drops from above the slime that won the previous point.
- **Scoring**: Points awarded when the ball lands on the opponent's side

## Project Structure

```
slime-volley/
├── game.py          # Core game logic and physics engine
├── agent.py         # DQN agent implementation with memory replay
├── model.py         # Neural network architecture and training logic
├── trainer.py       # Training loop and agent coordination
└── README.md        # This file
```

## Components

### Game Engine (`game.py`)
- `GameState`: Manages all game state variables (ball, paddles, scores)
- `SlimeVolleyball`: Main game controller with physics, rendering, and collision detection
- `Action`: Enum containing `LEFT`, `RIGHT`, `UP`, and `NOTHING`
- `SlimeVolleyball.reset()`: Initializes a game and returns observations for both models
- `SlimeVolleyball.step(player_one_action, player_two_action)`: Applies both actions and returns a `StepResult` containing the next states, emitted `GameEvent`s, `done`, and the round winner
- Player two's observation is mirrored so it always sees itself on the left,
  and its `LEFT`/`RIGHT` actions are mirrored the same way. One policy can play
  either side: `RIGHT` always means "toward the net".
- `SlimeVolleyball.reset(server=Player.PLAYER_ONE)` chooses who serves first.
- Features realistic paddle-ball collisions using physics-based calculations
- Visual elements include animated eyes that track the ball position

The simulation is headless by default, which is suitable for genetic-algorithm
evaluation. Rendering can be enabled with `SlimeVolleyball(render=True)`.

Example:

```python
from game import Action, SlimeVolleyball

game = SlimeVolleyball()
states = game.reset()

result = game.step(Action.RIGHT, Action.NOTHING)
player_one_state = result.states["player_one"]
player_two_state = result.states["player_two"]
if result.events:
    print(result.events)
```

Each step's events are one-step indicators. They include player hits,
crossings to either side of the net, and round wins. `done=True` indicates a
round ended; the ball is reset for the next round and scores are retained.
The game does not calculate numeric rewards. The DQN trainer (`trainer.py`)
maps events to per-step rewards. The genetic algorithm trains in PufferLib (see
below) and only counts points won.

### Genetic self-play

Training runs in PufferLib's Ocean `slimevolley` environment, a C port of
`slimevolleygym`, instead of `game.py`. `puffer_arena.py` puts every game of an
evaluation in one vectorized PufferLib environment, with one environment per
game, and both sides set to self-play (`num_agents=2`). Each step, `PolicyGroups`
runs each network once on all the slimes it controls in the batch, and games
that have finished are no longer computed. The worker pool splits the scheduled
games into `NUM_WORKERS * MATCH_TASKS_PER_WORKER` batches and plays them in
parallel, one batched PufferLib run per worker process.

Each game is played to `points_to_win` points (default 3). Each side earns 1
for every point it wins and `RETURN_OVER_NET_REWARD` (0.01) every time the ball
crosses the net from its side to the opponent's. A game's score is the agent's
share of the total (a 3-1 win with no returns scores 0.75), so ratings see the
margin, and agents that keep the ball in play earn a little credit before they
can score. A return is counted when the ball changes sides, so a serve or a
point reset never counts as one. Single-point games were too noisy: whoever
served often decided the result. PufferLib caps an episode
at 3000 ticks (30 ticks per second); a game that hits the cap ends with the
score so far, or a draw if no point was played. The four network outputs
map to PufferLib's `[forward, backward, jump]` buttons: `LEFT` (away from the
net), `RIGHT` (toward the net), `UP`, and `NOTHING`. Serves are launched with a
random velocity, and each batch gets a fresh seed.

### Kinematic bot benchmark

`kinematic_bot.py` is a scripted player that every generation also plays
against. When the ball is heading toward it, it simulates the ball's flight
with PufferLib's gravity, speed cap, walls and net to predict where the ball
comes down to slime height. If that point is on its side, it moves to stand
`return_offset` (0.3) units behind it, so the ball hits the net-facing side of
its dome and goes back over, and it jumps when the ball is within `jump_reach`
(2) units and below `jump_height` (9). Otherwise it waits at the middle of its
half. The first serve is handled the same way. It uses the same four actions as
the networks. Against 100 five-life episodes, it wins 78% of points against a
random network, 71% against the generation 1000 champion of the first PufferLib
run, and 11% against PufferLib's own built-in bot.

At the start of training the bot plays the five random bootstrap agents (40
games each) and its performance rating against them, with the random agents at
`1000`, is frozen as its rating for the whole run and saved in the training
state. It is added to the opponent pool every generation, so agents are always
rated partly against a target that never changes, and each generation logs the
champion's score against it on a `Benchmark:` line.

The bot is weighted heavily in fitness. Each agent plays it
`BOT_GAMES_MULTIPLIER` (4) times as many games as any other opponent, and
`performance_rating` weights those games so that together they count for
`BOT_FITNESS_SHARE` (half) of the agent's rating, with all the pool games
sharing the other half. Without the weighting the bot was one opponent in about
30 and the champions' score against it did not improve over 150 generations. It also takes part in the
final ladder round robin, labelled `bot`.

PufferLib's Ocean slime volley environment is not in the PyPI 3.0.0 release.
Build it from the `3.0` branch, and install `threadpoolctl`, which keeps NumPy's
matrix library single-threaded inside each worker (its threads make the tiny
policy multiplies about 150x slower). `NO_TRAIN=1` skips PufferLib's PyTorch C++
extension, which the genetic algorithm does not use and which fails to compile
on some macOS toolchains:

```
pip install threadpoolctl
NO_TRAIN=1 pip install --no-build-isolation \
    "pufferlib @ git+https://github.com/PufferAI/PufferLib.git@3b5c6046bb8b46685d62d151720025507e3418c2"
```

`train_agent_genetic.py` evolves a fixed-size population of `GeneticAgent`s.
Agents are never rated against each other. Each generation plays every agent
against the same set of frozen opponents, and those opponents' ratings never
change:

- **Rolling window:** the top 5 new agents from each of the last 5 generations.
- **Ladder:** the champion of every 10th generation, kept permanently (at most
  10 rungs; when it overflows, every other rung is dropped, keeping the oldest).
- **Bootstrap anchors:** generation 1 has no history, so it plays 5 random
  agents pinned at rating `1000`. They set the zero point and are then removed.
  Later ratings stay on that scale because each generation is rated against
  frozen ratings from earlier ones.

Ratings come from `performance_rating`: the single rating whose expected score
against the opponents equals the actual score. It does not depend on game
order, and a virtual draw against the average opponent keeps perfect records
finite. After the first pass, the top `finalist_count` agents (default 20) are
re-rated with `finalist_games_per_opponent` games (default 4) per opponent.
Survivors, the champion, and pool entrants are all chosen from these
finalists, so selection is not decided by two lucky games.

Each generation logs the champion's average score against the oldest and
newest ladder rungs next to the score its rating predicts. A large gap means the
rating scale has drifted.

### Measuring improvement: the ladder round robin

Ratings during training are measured against a moving opponent pool, so a flat
rating curve does not prove the agents stopped improving. To measure progress
directly, the champion of every 10th generation (and of the final generation)
is saved to `champions/ladder/gen_<generation>.pth`. When training finishes,
these rungs play a round robin against each other, the kinematic bot and a
fixed-seed random agent. Each pair plays `ladder_games_per_pair` games (default 20) with
sides alternating, scored the same way as training games.

Elo ratings are fitted to all of those results at once (a Bradley-Terry
maximum-likelihood fit, with one virtual draw per pair to keep perfect records
finite). The random agent is pinned at `1000`, so the numbers are comparable
across runs. Results are printed and written to
`champions/ladder/ladder_elo.csv`. If there are more than 25 rungs, an evenly
spaced subset is used, always including the first and the last.

To rate the saved ladder while training is still running, or after it stops
early, run the module with `--rate-ladder`.

Starting a new training run clears `champions/ladder/`; resuming keeps it.

The top `survival_rate` fraction survives unchanged, and the rest of the
population is refilled with mutated children of survivors.

`GeneticAgent` takes PufferLib's 12-value observation: x, y, vx, vy for the
agent's own slime, then the ball, then the opponent. Values are in world units
divided by 10. Each slime sees itself with its own side as positive x.
`SlimeVolleyball.get_puffer_observation` builds the same observation from
`game.py`, so trained agents can play in the pygame window. Checkpoints trained
before the switch to PufferLib used pixel observations and do not play
correctly with it.

`GeneticAgent.mutate(mutation_std=0.02)` returns a new agent. It deep-copies
the parent model and adds Gaussian noise to every weight; the parent is
unchanged.

### AI Agent (`agent.py`)
- Implements Deep Q-Learning with experience replay
- Uses epsilon-greedy exploration strategy with decay over time
- Memory buffer stores experiences for batch training
- Action space: [move_left, move_right, jump, stay]

### Neural Network (`model.py`)
- `QNet`: Deep Q-Network with 3 fully connected layers (10 → 256 → 256 → 4)
- Input: 10-dimensional state vector (ball position/velocity, player positions/velocities)
- Output: Q-values for 4 possible actions
- `QTrainer`: Handles model training with Adam optimizer and MSE loss

### Training Loop (`trainer.py`)
- Coordinates training between two agents
- Manages game steps, reward distribution, and memory updates
- Implements both short-term and long-term memory training

## State Representation

The game state is represented as a 10-dimensional vector:
- Ball position (x, y)
- Ball velocity (vx, vy)
- Player position (x, y) and vertical velocity
- Opponent position (x, y) and vertical velocity

## Reward System

- **+100**: Scoring a point (ball lands on opponent's side)
- **-100**: Losing a point (ball lands on your side)
- **+10**: Successfully hitting the ball
- **0**: All other actions

## Hyperparameters

- **Learning Rate**: 0.001
- **Discount Factor (γ)**: 0.9
- **Memory Size**: 100,000 experiences
- **Batch Size**: 1,000
- **Epsilon Decay**: Exponential decay from 0.5 to 0.2 over 100 games

## Installation

1. Ensure you have Python 3.x installed
2. Install required dependencies:
```bash
pip install pygame torch numpy
```

## Usage

### Training AI Agents
```bash
python trainer.py
```
This starts the training process where two AI agents play against each other and learn through self-play.

### Manual Play Mode
```bash
python game.py
```
This runs the game in manual control mode:
- **Player 1**: W (jump), A (left), D (right)
- **Player 2**: Up (jump), Left (left), Right (right)
- **R**: Reset ball
- **ESC**: Exit game

### Play Against the Genetic Champion

Each generation, training also overwrites `champions/top_agents/` with the
five highest-rated agents (`rank_<n>_gen_<generation>_rating_<rating>.pth`).
Any of them can be loaded with `GeneticAgent.load_model`.

After every generation the full training state (population, opponent pool,
ladder and ratings) is written to `champions/training_state.pth`, which is not
committed. To continue an interrupted run, pass `resume=True` to `train()` (or
`--resume` on the command line). To start a new run from saved top agents, pass
`seed_agents_directory=` (or `--seed-from champions/top_agents`): the population
is the top agents plus mutated copies of them, the generation count continues
from theirs, and they enter the opponent pool with their saved ratings.

After training has created the champion checkpoint, run:

```bash
python game.py champions/agent_genetic_gen_1000.pth
```

You control player one with W/A/D while the loaded genetic agent controls
player two using its mirrored observation.

## Training Process

The agents use a Deep Q-Learning approach:
1. **Experience Collection**: Agents take actions and observe rewards
2. **Memory Storage**: Experiences are stored in replay buffers
3. **Batch Training**: Neural networks are updated using random batches from memory
4. **Exploration vs Exploitation**: Epsilon-greedy strategy balances exploration and learned behavior
5. **Continuous Learning**: Agents improve through self-play over many games
