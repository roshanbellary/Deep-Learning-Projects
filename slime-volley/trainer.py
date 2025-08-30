from game import GameState, SlimeVolleyball
from agent import Agent
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self):
        self.game = SlimeVolleyball()
        self.agent_player = Agent( "player")
        self.agent_opponent = Agent( "opponent")

def _running_avg(xs):
    avg = []
    s = 0.0
    for i, v in enumerate(xs, 1):
        s += v
        avg.append(s / i)
    return avg

def plot_rewards(player_reward_collected, opponent_reward_collected, block=True):
    games = list(range(1, len(player_reward_collected) + 1))
    p_avg = _running_avg(player_reward_collected)
    o_avg = _running_avg(opponent_reward_collected)

    plt.clf()
    plt.subplot(1, 2, 1)
    # plt.plot(games, player_reward_collected, marker='o')
    plt.plot(games, p_avg, '-')
    plt.title("Player Rewards")

    plt.subplot(1, 2, 2)
    # plt.plot(games, opponent_reward_collected, marker='o')
    plt.plot(games, o_avg, '-')
    plt.title("Opponent Rewards")

    plt.pause(0.01)   # keeps one figure window open


def train():
    t = Trainer()
    player_reward_collected = []
    opponent_reward_collected = []

    player_reward_col = 0
    opponent_reward_col = 0

    max_player_reward = float("-inf")
    max_opp_reward = float("-inf")
    n_games = 0
    try:
        while True:
            state_old = t.game.get_state()
            player_move = t.agent_player.get_action(state_old)
            opponent_move = t.agent_opponent.get_action(state_old)

            player_reward, opponent_reward, done = t.game.play_step(player_move, opponent_move)

            player_reward_col += player_reward
            opponent_reward_col += opponent_reward

            state_new = t.game.get_state()

            t.agent_player.train_short_memory(state_old, player_move, player_reward, state_new, done)
            t.agent_opponent.train_short_memory(state_new, opponent_move, opponent_reward, state_new, done)

            t.agent_player.remember(state_old, player_move, player_reward, state_new, done)
            t.agent_opponent.remember(state_old, opponent_move, opponent_reward, state_new, done)

            if done:
                player_reward_collected.append(player_reward_col)
                opponent_reward_collected.append(opponent_reward_col)

                if player_reward_col > max_player_reward:
                    max_player_reward = player_reward_col
                    t.agent_player.save_model()
                
                if opponent_reward_col > max_opp_reward:
                    max_opp_reward = opponent_reward_col
                    t.agent_opponent.save_model()

                player_reward_col = 0
                opponent_reward_col = 0
                
                n_games += 1
                t.agent_player.n_games += 1
                t.agent_opponent.n_games += 1
                t.agent_player.train_long_memory()
                t.agent_opponent.train_long_memory()
                if n_games % 25 == 0:
                    plot_rewards(player_reward_collected, opponent_reward_collected, block=False)

    except KeyboardInterrupt:
        if player_reward_collected:
            plot_rewards(player_reward_collected, opponent_reward_collected, block=True)
        raise
if __name__ == "__main__":
    train()