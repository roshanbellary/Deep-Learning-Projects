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
- Features realistic paddle-ball collisions using physics-based calculations
- Visual elements include animated eyes that track the ball position

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
- **Player 1**: A/D (move), W/Space (jump)
- **Player 2**: Arrow keys (left/right/up)
- **R**: Reset ball
- **ESC**: Exit game

## Training Process

The agents use a Deep Q-Learning approach:
1. **Experience Collection**: Agents take actions and observe rewards
2. **Memory Storage**: Experiences are stored in replay buffers
3. **Batch Training**: Neural networks are updated using random batches from memory
4. **Exploration vs Exploitation**: Epsilon-greedy strategy balances exploration and learned behavior
5. **Continuous Learning**: Agents improve through self-play over many games

