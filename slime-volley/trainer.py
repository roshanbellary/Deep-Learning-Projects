from game import GameState, SlimeVolleyball
from agent import Agent

class Trainer:
    def __init__(self):
        self.game = SlimeVolleyball()
        self.agent_player = Agent( "player")
        self.agent_opponent = Agent( "opponent")


def train():
    t = Trainer()
    while True:
        state_old = t.game.get_state()
        player_move = t.agent_player.get_action(state_old)
        opponent_move = t.agent_opponent.get_action(state_old)

        player_reward, opponent_reward, done = t.game.play_step(player_move, opponent_move)

        state_new = t.game.get_state()

        t.agent_player.train_short_memory(state_old, player_move, player_reward, state_new, done)
        t.agent_opponent.train_short_memory(state_new, opponent_move, opponent_reward, state_new, done)

        t.agent_player.remember(state_old, player_move, player_reward, state_new, done)
        t.agent_opponent.remember(state_old, opponent_move, opponent_reward, state_new, done)

        if done:
            t.agent_player.n_games += 1
            t.agent_opponent.n_games += 1
            t.agent_player.train_long_memory()
            t.agent_opponent.train_long_memory()

if __name__ == "__main__":
    train()