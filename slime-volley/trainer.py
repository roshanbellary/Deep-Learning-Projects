from game import GameEvent, Player, SlimeVolleyball
from agent import Agent
import matplotlib.pyplot as plt


HIT_REWARD = 2.0
SIDE_CROSSING_REWARD = 4.0
ROUND_WIN_REWARD = 100.0
ROUND_LOSS_REWARD = -100.0


def _event_rewards(events, hit_counts):
    player_reward = 0.0
    opponent_reward = 0.0

    if (
        GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE in events
        or GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE in events
    ):
        hit_counts[Player.PLAYER_ONE] = 0
        hit_counts[Player.PLAYER_TWO] = 0

    if GameEvent.PLAYER_ONE_HIT in events:
        count = hit_counts[Player.PLAYER_ONE]
        player_reward += HIT_REWARD / (2 ** count)
        hit_counts[Player.PLAYER_ONE] += 1
    if GameEvent.PLAYER_TWO_HIT in events:
        count = hit_counts[Player.PLAYER_TWO]
        opponent_reward += HIT_REWARD / (2 ** count)
        hit_counts[Player.PLAYER_TWO] += 1
    if GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE in events:
        player_reward += SIDE_CROSSING_REWARD
    if GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE in events:
        opponent_reward += SIDE_CROSSING_REWARD
    if GameEvent.PLAYER_ONE_WON_ROUND in events:
        player_reward += ROUND_WIN_REWARD
        opponent_reward += ROUND_LOSS_REWARD
    if GameEvent.PLAYER_TWO_WON_ROUND in events:
        opponent_reward += ROUND_WIN_REWARD
        player_reward += ROUND_LOSS_REWARD

    return player_reward, opponent_reward

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
    plt.plot(games, p_avg, '-')
    plt.title("Player Rewards")

    plt.subplot(1, 2, 2)
    plt.plot(games, o_avg, '-')
    plt.title("Opponent Rewards")

    plt.pause(0.01)


def train():
    t = Trainer()
    player_reward_collected = []
    opponent_reward_collected = []

    player_reward_col = 0
    opponent_reward_col = 0

    max_player_reward = float("-inf")
    max_opp_reward = float("-inf")
    n_games = 0
    hit_counts = {
        Player.PLAYER_ONE: 0,
        Player.PLAYER_TWO: 0,
    }
    try:
        while True:
            states_old = t.game.get_states()
            state_old = states_old[Player.PLAYER_ONE.value]
            opponent_state_old = states_old[Player.PLAYER_TWO.value]
            player_move = t.agent_player.get_action(state_old)
            opponent_move = t.agent_opponent.get_action(opponent_state_old)

            result = t.game.play_step(player_move, opponent_move)
            player_reward, opponent_reward = _event_rewards(result.events, hit_counts)
            done = result.done

            player_reward_col += player_reward
            opponent_reward_col += opponent_reward

            state_new = result.states[Player.PLAYER_ONE.value]
            opponent_state_new = result.states[Player.PLAYER_TWO.value]

            t.agent_player.train_short_memory(state_old, player_move, player_reward, state_new, done)
            t.agent_opponent.train_short_memory(opponent_state_old, opponent_move, opponent_reward, opponent_state_new, done)

            t.agent_player.remember(state_old, player_move, player_reward, state_new, done)
            t.agent_opponent.remember(opponent_state_old, opponent_move, opponent_reward, opponent_state_new, done)

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
                hit_counts[Player.PLAYER_ONE] = 0
                hit_counts[Player.PLAYER_TWO] = 0
                
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
