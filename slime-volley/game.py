from dataclasses import dataclass
from enum import Enum
import math
import random
from pathlib import Path
import sys
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

try:
    import pygame
except ImportError:
    pygame = None


class Action(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    NOTHING = "nothing"


ActionState = Action
GameAction = Action


class Player(Enum):
    PLAYER_ONE = "player_one"
    PLAYER_TWO = "player_two"


class GameEvent(Enum):
    PLAYER_ONE_HIT = "player_one_hit"
    PLAYER_TWO_HIT = "player_two_hit"
    BALL_CROSSED_TO_PLAYER_ONE_SIDE = "ball_crossed_to_player_one_side"
    BALL_CROSSED_TO_PLAYER_TWO_SIDE = "ball_crossed_to_player_two_side"
    PLAYER_ONE_WON_ROUND = "player_one_won_round"
    PLAYER_TWO_WON_ROUND = "player_two_won_round"

    PLAYER_HIT = PLAYER_ONE_HIT
    OPPONENT_HIT = PLAYER_TWO_HIT
    PLAYER_GAME_OVER = PLAYER_ONE_WON_ROUND
    OPPONENT_GAME_OVER = PLAYER_TWO_WON_ROUND


@dataclass(frozen=True)
class StepResult:
    states: Dict[str, List[float]]
    events: FrozenSet[GameEvent]
    done: bool
    winner: Optional[Player] = None

    @property
    def event_flags(self) -> Dict[GameEvent, bool]:
        return {event: event in self.events for event in GameEvent}

    def __iter__(self):
        yield self.states
        yield self.events
        yield self.done


class GameState:
    def __init__(self):
        self.SCREEN_WIDTH = 750
        self.SCREEN_HEIGHT = 370
        self.GROUND_Y = 300

        self.PADDLE_RADIUS = 40
        self.BALL_RADIUS = 10
        self.BALL_SPEED = 4
        self.BALL_MAX_SPEED = 12
        self.MOVEMENT_SPEED = 2
        self.GRAVITY = 0.4
        self.JUMP_FORCE = -6

        self.NET_X = 375
        self.NET_WIDTH = 5
        self.NET_HEIGHT = 50

        self.collision_coefficient = 1
        self.collision_margin = 0

        self.player_x = 125
        self.player_y = self.GROUND_Y - self.PADDLE_RADIUS
        self.player_vy = 0
        self.opponent_x = 562.5
        self.opponent_y = self.GROUND_Y - self.PADDLE_RADIUS
        self.opponent_vy = 0

        self.ball_x = 375
        self.ball_y = 100
        self.ball_vx = 0
        self.ball_vy = 0

        self.player_score = 0
        self.opponent_score = 0

ActionInput = Union[Action, str, int, List[int], Tuple[int, ...]]
class SlimeVolleyball:
    def __init__(self, render: bool = False, serve_sideways_jitter: float = 0.0):
        self.render_enabled = render
        self.serve_sideways_jitter = serve_sideways_jitter
        self.screen = None
        self.clock = None
        self.running = True

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)

        self.state = GameState()
        self.last_result: Optional[StepResult] = None
        self._last_ball_side = self._ball_side()
        self.reset_ball(Player.PLAYER_ONE)

        if self.render_enabled:
            self._initialize_renderer()

    def _initialize_renderer(self):
        if pygame is None:
            raise RuntimeError(
                "pygame is required for rendering. Install it in the project's virtual environment."
            )
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.state.SCREEN_WIDTH, self.state.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Slime Volleyball")
        self.clock = pygame.time.Clock()

    def initialize_game(self) -> Dict[str, List[float]]:
        return self.reset()

    def reset(self) -> Dict[str, List[float]]:
        self.state = GameState()
        self.running = True
        self.last_result = None
        self.reset_ball(Player.PLAYER_ONE)
        return self.get_states()

    def reset_game(self) -> Dict[str, List[float]]:
        return self.reset()

    def reset_ball(self, winner: Union[Player, str] = Player.PLAYER_ONE):
        player = self._coerce_player(winner)
        if player is Player.PLAYER_ONE:
            self.state.ball_x = self.state.player_x + self.state.PADDLE_RADIUS
            self.state.ball_y = self.state.player_y - self.state.BALL_RADIUS - 10
        else:
            self.state.ball_x = self.state.opponent_x + self.state.PADDLE_RADIUS
            self.state.ball_y = self.state.opponent_y - self.state.BALL_RADIUS - 10
        self.state.ball_vx = random.uniform(-self.serve_sideways_jitter, self.serve_sideways_jitter)
        self.state.ball_vy = -self.state.BALL_SPEED
        self._last_ball_side = self._ball_side()

    @staticmethod
    def _coerce_player(player: Union[Player, str]) -> Player:
        if isinstance(player, Player):
            return player
        aliases = {
            "player": Player.PLAYER_ONE,
            "opponent": Player.PLAYER_TWO,
            "player_one": Player.PLAYER_ONE,
            "player_two": Player.PLAYER_TWO,
        }
        try:
            return aliases[player.lower()]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"Unknown player: {player!r}") from exc

    @staticmethod
    def _coerce_action(action: ActionInput) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, str):
            aliases = {
                "left": Action.LEFT,
                "right": Action.RIGHT,
                "up": Action.UP,
                "nothing": Action.NOTHING,
                "none": Action.NOTHING,
                "stay": Action.NOTHING,
            }
            try:
                return aliases[action.lower()]
            except KeyError as exc:
                raise ValueError(f"Unknown action: {action!r}") from exc
        if isinstance(action, int):
            actions = (Action.LEFT, Action.RIGHT, Action.UP, Action.NOTHING)
            try:
                return actions[action]
            except IndexError as exc:
                raise ValueError(f"Action index must be between 0 and 3: {action}") from exc
        if isinstance(action, (list, tuple)):
            if len(action) != 4 or sum(value == 1 for value in action) != 1:
                raise ValueError("One-hot actions must contain exactly one 1 across four values")
            return SlimeVolleyball._coerce_action(action.index(1))
        raise TypeError(f"Unsupported action type: {type(action).__name__}")

    def move_player(self, direction: int):
        new_x = self.state.player_x + direction * self.state.MOVEMENT_SPEED
        if 0 <= new_x <= self.state.NET_X - self.state.PADDLE_RADIUS * 2:
            self.state.player_x = new_x

    def jump_player(self):
        if self.state.player_y >= self.state.GROUND_Y - self.state.PADDLE_RADIUS:
            self.state.player_vy = self.state.JUMP_FORCE

    def move_opponent(self, direction: int):
        new_x = self.state.opponent_x + direction * self.state.MOVEMENT_SPEED
        if (
            self.state.NET_X + self.state.NET_WIDTH <= new_x
            <= self.state.SCREEN_WIDTH - self.state.PADDLE_RADIUS * 2
        ):
            self.state.opponent_x = new_x

    def jump_opponent(self):
        if self.state.opponent_y >= self.state.GROUND_Y - self.state.PADDLE_RADIUS:
            self.state.opponent_vy = self.state.JUMP_FORCE

    def _play_player_action(self, action: ActionInput):
        action = self._coerce_action(action)
        if action is Action.LEFT:
            self.move_player(-1)
        elif action is Action.RIGHT:
            self.move_player(1)
        elif action is Action.UP:
            self.jump_player()

    def _play_opponent_action(self, action: ActionInput):
        action = self._coerce_action(action)
        if action is Action.LEFT:
            self.move_opponent(-1)
        elif action is Action.RIGHT:
            self.move_opponent(1)
        elif action is Action.UP:
            self.jump_opponent()

    def step(self, player_one_action: ActionInput, player_two_action: ActionInput) -> StepResult:
        self._play_player_action(player_one_action)
        self._play_opponent_action(player_two_action)
        events, done, winner = self._advance()
        result = StepResult(self.get_states(), frozenset(events), done, winner)
        self.last_result = result
        return result

    def play_step(self, player_action: ActionInput, opponent_action: ActionInput):
        return self.step(player_action, opponent_action)

    def _advance(self):
        self.update_paddle_physics()
        return self._update_ball_physics()

    def update(self):
        events, done, winner = self._advance()
        self.last_result = StepResult(self.get_states(), frozenset(events), done, winner)
        return frozenset(events), done, winner

    def update_paddle_physics(self):
        ground_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
        self.state.player_y += self.state.player_vy
        self.state.player_vy += self.state.GRAVITY
        if self.state.player_y > ground_y:
            self.state.player_y = ground_y
            self.state.player_vy = 0

        self.state.opponent_y += self.state.opponent_vy
        self.state.opponent_vy += self.state.GRAVITY
        if self.state.opponent_y > ground_y:
            self.state.opponent_y = ground_y
            self.state.opponent_vy = 0

    def _ball_side(self) -> Player:
        return Player.PLAYER_ONE if self.state.ball_x < self.state.NET_X else Player.PLAYER_TWO

    def _update_ball_physics(self):
        events = set()
        done = False
        winner = None
        previous_side = self._ball_side()

        self.state.ball_x += self.state.ball_vx
        self.state.ball_y += self.state.ball_vy
        self.state.ball_vy += self.state.GRAVITY

        if self.state.ball_x <= self.state.BALL_RADIUS or self.state.ball_x >= self.state.SCREEN_WIDTH - self.state.BALL_RADIUS:
            self.state.ball_vx *= -1
            self.state.ball_x = max(self.state.BALL_RADIUS, min(self.state.SCREEN_WIDTH - self.state.BALL_RADIUS, self.state.ball_x))

        if self.state.ball_y <= self.state.BALL_RADIUS:
            self.state.ball_vy *= -self.state.collision_coefficient
            self.state.ball_y = self.state.BALL_RADIUS

        current_side = self._ball_side()
        if current_side is not previous_side:
            if current_side is Player.PLAYER_ONE:
                events.add(GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE)
            else:
                events.add(GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE)
        self._last_ball_side = current_side

        if self.state.ball_y >= self.state.GROUND_Y - self.state.BALL_RADIUS:
            self.state.ball_y = self.state.GROUND_Y - self.state.BALL_RADIUS
            if current_side is Player.PLAYER_ONE:
                winner = Player.PLAYER_TWO
                self.state.opponent_score += 1
                events.add(GameEvent.PLAYER_TWO_WON_ROUND)
            else:
                winner = Player.PLAYER_ONE
                self.state.player_score += 1
                events.add(GameEvent.PLAYER_ONE_WON_ROUND)

            done = True
            self.reset_paddle_positions()
            self.reset_ball(winner)
            return events, done, winner

        if (
            self.state.NET_X - self.state.BALL_RADIUS <= self.state.ball_x <= self.state.NET_X + self.state.NET_WIDTH + self.state.BALL_RADIUS
            and self.state.ball_y >= self.state.GROUND_Y - self.state.NET_HEIGHT - self.state.BALL_RADIUS
        ):
            self.state.ball_vx *= -1

        if self._collides_with_paddle(Player.PLAYER_ONE):
            events.add(GameEvent.PLAYER_ONE_HIT)

        if self._collides_with_paddle(Player.PLAYER_TWO):
            events.add(GameEvent.PLAYER_TWO_HIT)

        return events, done, winner

    def update_ball_physics(self):
        events, done, winner = self._update_ball_physics()
        self.last_result = StepResult(self.get_states(), frozenset(events), done, winner)
        return frozenset(events), done, winner

    def _collides_with_paddle(self, player: Player) -> bool:
        if player is Player.PLAYER_ONE:
            paddle_x, paddle_y, paddle_vy = self.state.player_x, self.state.player_y, self.state.player_vy
        else:
            paddle_x, paddle_y, paddle_vy = self.state.opponent_x, self.state.opponent_y, self.state.opponent_vy

        center_x = paddle_x + self.state.PADDLE_RADIUS
        center_y = paddle_y + self.state.PADDLE_RADIUS
        distance = math.hypot(self.state.ball_x - center_x, self.state.ball_y - center_y)
        collision_radius = (self.state.PADDLE_RADIUS + self.state.BALL_RADIUS) * (1 + self.state.collision_margin)
        if distance > collision_radius or self.state.ball_y > center_y:
            return False

        distance_vector = (center_x - self.state.ball_x, center_y - self.state.ball_y)
        relative_velocity = (self.state.ball_vx, self.state.ball_vy - paddle_vy)
        if distance_vector[0] * relative_velocity[0] + distance_vector[1] * relative_velocity[1] <= 0:
            return False

        if distance == 0:
            normal_x, normal_y = (1.0, -1.0 / math.sqrt(2))
        else:
            normal_x = (self.state.ball_x - center_x) / distance
            normal_y = (self.state.ball_y - center_y) / distance
        collision_distance = self.state.PADDLE_RADIUS + self.state.BALL_RADIUS
        self.state.ball_x = center_x + normal_x * collision_distance
        self.state.ball_y = center_y + normal_y * collision_distance
        self.state.ball_vx, self.state.ball_vy = self.calculate_ball_velocity_after_collision(
            center_x, center_y, self.state.ball_x, self.state.ball_y,
            self.state.ball_vx, self.state.ball_vy, 0, paddle_vy,
        )
        return True

    def calculate_ball_velocity_after_collision(self, paddle_x, paddle_y, ball_x, ball_y, ball_vx, ball_vy, paddle_vx, paddle_vy):
        distance_vector = (paddle_x - ball_x, paddle_y - ball_y)
        v_ref = (ball_vx - paddle_vx, ball_vy - paddle_vy)
        denominator = distance_vector[0] ** 2 + distance_vector[1] ** 2
        if denominator == 0:
            return ball_vx, -abs(ball_vy)
        dot_product = distance_vector[0] * v_ref[0] + distance_vector[1] * v_ref[1]
        new_velocity = (
            v_ref[0] - 2 * (dot_product / denominator) * distance_vector[0] + paddle_vx,
            v_ref[1] - 2 * (dot_product / denominator) * distance_vector[1] + paddle_vy,
        )
        new_velocity = tuple(value * self.state.collision_coefficient for value in new_velocity)
        magnitude = math.hypot(*new_velocity)
        if magnitude > self.state.BALL_MAX_SPEED:
            scale = self.state.BALL_MAX_SPEED / magnitude
            new_velocity = (new_velocity[0] * scale, new_velocity[1] * scale)
        return new_velocity

    def get_state(self, player: Union[Player, str] = Player.PLAYER_ONE) -> List[float]:
        player = self._coerce_player(player)
        values = [
            self.state.ball_x, self.state.ball_y, self.state.ball_vx, self.state.ball_vy,
            self.state.player_x, self.state.player_y, self.state.player_vy,
            self.state.opponent_x, self.state.opponent_y, self.state.opponent_vy,
        ]
        if player is Player.PLAYER_ONE:
            return values

        radius = self.state.PADDLE_RADIUS
        return [
            self.state.SCREEN_WIDTH - self.state.ball_x, self.state.ball_y, -self.state.ball_vx, self.state.ball_vy,
            self.state.SCREEN_WIDTH - self.state.opponent_x - 2 * radius, self.state.opponent_y, self.state.opponent_vy,
            self.state.SCREEN_WIDTH - self.state.player_x - 2 * radius, self.state.player_y, self.state.player_vy,
        ]

    def get_states(self) -> Dict[str, List[float]]:
        return {
            Player.PLAYER_ONE.value: self.get_state(Player.PLAYER_ONE),
            Player.PLAYER_TWO.value: self.get_state(Player.PLAYER_TWO),
        }

    def handle_input(self, control_opponent: bool = True):
        if pygame is None:
            raise RuntimeError("pygame is required for keyboard input")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.move_player(-1)
        if keys[pygame.K_d]:
            self.move_player(1)
        if keys[pygame.K_w]:
            self.jump_player()
        if control_opponent:
            if keys[pygame.K_LEFT]:
                self.move_opponent(-1)
            if keys[pygame.K_RIGHT]:
                self.move_opponent(1)
            if keys[pygame.K_UP]:
                self.jump_opponent()
        if keys[pygame.K_r]:
            self.reset_ball(Player.PLAYER_ONE)

    @staticmethod
    def _agent_action(agent, state: List[float]) -> Action:
        import torch

        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            action_index = agent.move(state_tensor)
        if hasattr(action_index, "item"):
            action_index = action_index.item()
        return list(Action)[int(action_index)]

    def play_against_agent(self, agent):
        if pygame is None:
            raise RuntimeError("pygame is required to play against an agent")
        if self.screen is None:
            self._initialize_renderer()
        agent.model.eval()
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            self.handle_input(control_opponent=False)
            ai_action = self._agent_action(
                agent,
                self.get_state(Player.PLAYER_TWO),
            )
            self.step(Action.NOTHING, ai_action)
            self.render()
            self.clock.tick(60)
        pygame.quit()

    def play_against_champion(self, champion_path=None):
        if champion_path is None:
            champion_path = (
                Path(__file__).resolve().parent
                / "champions"
                / "agent_genetic_gen_1000.pth"
            )
        try:
            from .agent_genetic import GeneticAgent
        except ImportError:
            from agent_genetic import GeneticAgent
        champion = GeneticAgent.load_model(champion_path)
        self.play_against_agent(champion)

    def reset_paddle_positions(self):
        self.state.player_x = self.state.SCREEN_WIDTH / 4 - self.state.PADDLE_RADIUS
        self.state.player_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
        self.state.player_vy = 0
        self.state.opponent_x = self.state.SCREEN_WIDTH * 3 / 4 - self.state.PADDLE_RADIUS
        self.state.opponent_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
        self.state.opponent_vy = 0

    def draw_semicircle(self, surface, color, center, radius, top_half=True):
        temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, color, (radius, radius), radius)
        if top_half:
            pygame.draw.rect(temp_surface, (0, 0, 0, 0), (0, radius, radius * 2, radius))
        else:
            pygame.draw.rect(temp_surface, (0, 0, 0, 0), (0, 0, radius * 2, radius))
        surface.blit(temp_surface, (center[0] - radius, center[1] - radius))

    def draw_paddle_eyes(self, surface, paddle_center, paddle_radius, eye_color, ball_x, ball_y, which_paddle):
        angle = math.atan2(ball_y - paddle_center[1], ball_x - paddle_center[0])
        eye_offset = paddle_radius * 0.3
        eye_x = paddle_center[0] + (eye_offset if which_paddle == "player" else -eye_offset)
        eye_y = paddle_center[1] - eye_offset * 1.5
        eye_radius = paddle_radius * 0.2
        pygame.draw.circle(surface, self.WHITE, (int(eye_x), int(eye_y)), int(eye_radius))
        pupil_offset = eye_radius * 0.4
        pupil_x = eye_x + math.cos(angle) * pupil_offset
        pupil_y = eye_y + math.sin(angle) * pupil_offset
        pygame.draw.circle(surface, eye_color, (int(pupil_x), int(pupil_y)), int(eye_radius * 0.6))
        pygame.draw.circle(surface, eye_color, (int(eye_x), int(eye_y)), int(eye_radius), 2)

    def render(self):
        if pygame is None:
            raise RuntimeError("pygame is required for rendering")
        if self.screen is None:
            self._initialize_renderer()
        self.screen.fill(self.BLACK)
        pygame.draw.rect(self.screen, self.GREEN, (0, self.state.GROUND_Y, self.state.SCREEN_WIDTH, self.state.SCREEN_HEIGHT - self.state.GROUND_Y))
        pygame.draw.rect(self.screen, self.WHITE, (self.state.NET_X, self.state.GROUND_Y - self.state.NET_HEIGHT, self.state.NET_WIDTH, self.state.NET_HEIGHT))
        self.draw_semicircle(self.screen, self.BLUE, (int(self.state.player_x + self.state.PADDLE_RADIUS), int(self.state.player_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, True)
        self.draw_semicircle(self.screen, self.RED, (int(self.state.opponent_x + self.state.PADDLE_RADIUS), int(self.state.opponent_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, True)
        self.draw_paddle_eyes(self.screen, (int(self.state.player_x + self.state.PADDLE_RADIUS), int(self.state.player_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, self.BLACK, self.state.ball_x, self.state.ball_y, "player")
        self.draw_paddle_eyes(self.screen, (int(self.state.opponent_x + self.state.PADDLE_RADIUS), int(self.state.opponent_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, self.BLACK, self.state.ball_x, self.state.ball_y, "opponent")
        pygame.draw.circle(self.screen, self.WHITE, (int(self.state.ball_x), int(self.state.ball_y)), self.state.BALL_RADIUS)
        font = pygame.font.Font(None, 36)
        self.screen.blit(font.render(f"Player: {self.state.player_score}", True, self.BLUE), (10, 10))
        self.screen.blit(font.render(f"Opponent: {self.state.opponent_score}", True, self.RED), (self.state.SCREEN_WIDTH - 150, 10))
        pygame.display.flip()

    def run(self, manual_control=True):
        if pygame is None:
            raise RuntimeError("pygame is required to run the manual game")
        if self.screen is None:
            self._initialize_renderer()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
            if manual_control:
                self.handle_input()
            self.update()
            self.render()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    game = SlimeVolleyball(render=True)
    if len(sys.argv) > 1:
        game.play_against_champion(sys.argv[1])
    else:
        game.run()
