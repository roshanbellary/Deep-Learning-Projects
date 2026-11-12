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
The game does not calculate numeric rewards. The training layer maps events to
points: `+2` for the first hit while the ball remains on a side, `+4` for
sending the ball to the other side, and `+100` for winning a round.
Consecutive hits on the same side decay by half each time (`+2`, `+1`,
`+0.5`, ...). The training-layer hit counter resets when the ball crosses the
net or a new round begins.

### Genetic self-play

`train_agent_genetic.py` provides `_simulate_games(agent1, agent2, games=2)`.
By default it plays two rounds total, swapping court sides between rounds, and uses the mirrored
observations from `game.get_states()`, and scores each agent from hit,
crossing, and round-win events. It returns aggregate scores, win/draw counts,
and per-round records. The dynamic reward totals determine each matchup's
winner, and Elo is the sole population-ranking metric. Elo values on the two
`GeneticAgent` instances are updated after each round. Selected survivors are
reset to Elo `1000` before the next generation. Matchups are evaluated with a four-process
multiprocessing pool; fitness and Elo updates are applied in the parent
process after worker results return.

`GeneticAgent.mutate(mutation_rate=0.01, mutation_std=0.1)` returns a new
agent. It deep-copies the parent model, samples an independent Bernoulli mask
for every parameter element, and adds Gaussian noise to masked elements only;
the parent is unchanged.

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
