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


ARENA_WIDTH_UNITS = 48.0
SCREEN_WIDTH_PIXELS = 750
SCREEN_HEIGHT_PIXELS = 400
GROUND_Y_PIXELS = 350
PIXELS_PER_UNIT = SCREEN_WIDTH_PIXELS / ARENA_WIDTH_UNITS
FRAMES_PER_SECOND = 60
SCORE_MARGIN_PIXELS = 10

SLIME_RADIUS_UNITS = 1.5
BALL_RADIUS_UNITS = 0.5
NET_WIDTH_UNITS = 1.0
NET_HEIGHT_UNITS = 3.5
SERVE_HEIGHT_UNITS = 6.0

GRAVITY_UNITS_PER_SECOND_SQUARED = 29.4
SLIME_JUMP_SPEED_UNITS_PER_SECOND = 13.5
SLIME_RUN_SPEED_UNITS_PER_SECOND = 17.5
BALL_MAX_SPEED_UNITS_PER_SECOND = 22.5
BALL_RESTITUTION = 1.0


def units_to_pixels(distance_units: float) -> float:
    return distance_units * PIXELS_PER_UNIT


def speed_to_pixels_per_frame(speed_units_per_second: float) -> float:
    return speed_units_per_second * PIXELS_PER_UNIT / FRAMES_PER_SECOND


def acceleration_to_pixels_per_frame_squared(acceleration_units_per_second_squared: float) -> float:
    return acceleration_units_per_second_squared * PIXELS_PER_UNIT / FRAMES_PER_SECOND ** 2


def clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(lower_bound, min(upper_bound, value))


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


HIT_EVENT_BY_PLAYER = {
    Player.PLAYER_ONE: GameEvent.PLAYER_ONE_HIT,
    Player.PLAYER_TWO: GameEvent.PLAYER_TWO_HIT,
}
CROSSING_EVENT_BY_DESTINATION = {
    Player.PLAYER_ONE: GameEvent.BALL_CROSSED_TO_PLAYER_ONE_SIDE,
    Player.PLAYER_TWO: GameEvent.BALL_CROSSED_TO_PLAYER_TWO_SIDE,
}
ROUND_WIN_EVENT_BY_PLAYER = {
    Player.PLAYER_ONE: GameEvent.PLAYER_ONE_WON_ROUND,
    Player.PLAYER_TWO: GameEvent.PLAYER_TWO_WON_ROUND,
}
STATE_PREFIX_BY_PLAYER = {
    Player.PLAYER_ONE: "player",
    Player.PLAYER_TWO: "opponent",
}
LEFT_DIRECTION = -1
RIGHT_DIRECTION = 1
NO_DIRECTION = 0
VIEW_MIRROR_SIGN_BY_PLAYER = {
    Player.PLAYER_ONE: 1,
    Player.PLAYER_TWO: -1,
}


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


@dataclass(frozen=True)
class Contact:
    normal_x: float
    normal_y: float
    penetration: float


class GameState:
    def __init__(self):
        self.SCREEN_WIDTH = SCREEN_WIDTH_PIXELS
        self.SCREEN_HEIGHT = SCREEN_HEIGHT_PIXELS
        self.GROUND_Y = GROUND_Y_PIXELS

        self.PADDLE_RADIUS = units_to_pixels(SLIME_RADIUS_UNITS)
        self.BALL_RADIUS = units_to_pixels(BALL_RADIUS_UNITS)
        self.BALL_MAX_SPEED = speed_to_pixels_per_frame(BALL_MAX_SPEED_UNITS_PER_SECOND)
        self.MOVEMENT_SPEED = speed_to_pixels_per_frame(SLIME_RUN_SPEED_UNITS_PER_SECOND)
        self.GRAVITY = acceleration_to_pixels_per_frame_squared(GRAVITY_UNITS_PER_SECOND_SQUARED)
        self.JUMP_FORCE = -speed_to_pixels_per_frame(SLIME_JUMP_SPEED_UNITS_PER_SECOND)
        self.SERVE_HEIGHT = units_to_pixels(SERVE_HEIGHT_UNITS)

        self.NET_WIDTH = units_to_pixels(NET_WIDTH_UNITS)
        self.NET_HEIGHT = units_to_pixels(NET_HEIGHT_UNITS)
        self.NET_X = self.SCREEN_WIDTH / 2 - self.NET_WIDTH / 2

        self.collision_coefficient = BALL_RESTITUTION

        self.place_slimes_at_start()

        self.ball_x = self.SCREEN_WIDTH / 2
        self.ball_y = self.GROUND_Y - self.SERVE_HEIGHT
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        self.player_score = 0
        self.opponent_score = 0

    @property
    def NET_CENTER_X(self) -> float:
        return self.SCREEN_WIDTH / 2

    @property
    def SLIME_GROUND_TOP_Y(self) -> float:
        return self.GROUND_Y - self.PADDLE_RADIUS

    def place_slimes_at_start(self):
        self.player_x = self.SCREEN_WIDTH / 4 - self.PADDLE_RADIUS
        self.player_y = self.SLIME_GROUND_TOP_Y
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.opponent_x = self.SCREEN_WIDTH * 3 / 4 - self.PADDLE_RADIUS
        self.opponent_y = self.SLIME_GROUND_TOP_Y
        self.opponent_vx = 0.0
        self.opponent_vy = 0.0


ActionInput = Union[Action, str, int, List[int], Tuple[int, ...]]


class SlimeVolleyball:
    def __init__(self, render: bool = False, serve_sideways_jitter: float = 0.0):
        self.render_enabled = render
        self.serve_sideways_jitter = serve_sideways_jitter
        self.screen = None
        self.clock = None
        self.score_font = None
        self.running = True

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)

        self.state = GameState()
        self.last_result: Optional[StepResult] = None
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
        self.score_font = pygame.font.Font(None, 36)

    def initialize_game(self) -> Dict[str, List[float]]:
        return self.reset()

    def reset(self, server: Union[Player, str] = Player.PLAYER_ONE) -> Dict[str, List[float]]:
        self.state = GameState()
        self.running = True
        self.last_result = None
        self.reset_ball(server)
        return self.get_states()

    def reset_game(self) -> Dict[str, List[float]]:
        return self.reset()

    def reset_ball(self, server: Union[Player, str] = Player.PLAYER_ONE):
        server = self._coerce_player(server)
        server_center_x, _ = self._slime_center(server)
        self.state.ball_x = server_center_x
        self.state.ball_y = self.state.GROUND_Y - self.state.SERVE_HEIGHT
        self.state.ball_vx = random.uniform(-self.serve_sideways_jitter, self.serve_sideways_jitter)
        self.state.ball_vy = 0.0

    def reset_paddle_positions(self):
        self.state.place_slimes_at_start()

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

    def _get_slime_attribute(self, player: Player, attribute: str) -> float:
        return getattr(self.state, f"{STATE_PREFIX_BY_PLAYER[player]}_{attribute}")

    def _set_slime_attribute(self, player: Player, attribute: str, value: float):
        setattr(self.state, f"{STATE_PREFIX_BY_PLAYER[player]}_{attribute}", value)

    def _slime_center(self, player: Player) -> Tuple[float, float]:
        radius = self.state.PADDLE_RADIUS
        return (
            self._get_slime_attribute(player, "x") + radius,
            self._get_slime_attribute(player, "y") + radius,
        )

    def _slime_x_bounds(self, player: Player) -> Tuple[float, float]:
        diameter = 2 * self.state.PADDLE_RADIUS
        if player is Player.PLAYER_ONE:
            return 0.0, self.state.NET_X - diameter
        return self.state.NET_X + self.state.NET_WIDTH, self.state.SCREEN_WIDTH - diameter

    def _is_grounded(self, player: Player) -> bool:
        return self._get_slime_attribute(player, "y") >= self.state.SLIME_GROUND_TOP_Y

    def _steer_slime(self, player: Player, horizontal_direction: int):
        self._set_slime_attribute(player, "vx", horizontal_direction * self.state.MOVEMENT_SPEED)

    def _jump_slime(self, player: Player):
        if self._is_grounded(player):
            self._set_slime_attribute(player, "vy", self.state.JUMP_FORCE)

    def _apply_action(self, player: Player, action: ActionInput):
        action = self._coerce_action(action)
        direction_in_own_view = {
            Action.LEFT: LEFT_DIRECTION,
            Action.RIGHT: RIGHT_DIRECTION,
        }.get(action, NO_DIRECTION)
        self._steer_slime(player, direction_in_own_view * VIEW_MIRROR_SIGN_BY_PLAYER[player])
        if action is Action.UP:
            self._jump_slime(player)

    def move_player(self, direction: int):
        self._steer_slime(Player.PLAYER_ONE, direction)

    def jump_player(self):
        self._jump_slime(Player.PLAYER_ONE)

    def move_opponent(self, direction: int):
        self._steer_slime(Player.PLAYER_TWO, direction)

    def jump_opponent(self):
        self._jump_slime(Player.PLAYER_TWO)

    def step(self, player_one_action: ActionInput, player_two_action: ActionInput) -> StepResult:
        self._apply_action(Player.PLAYER_ONE, player_one_action)
        self._apply_action(Player.PLAYER_TWO, player_two_action)
        return self._advance()

    def play_step(self, player_action: ActionInput, opponent_action: ActionInput):
        return self.step(player_action, opponent_action)

    def _advance(self) -> StepResult:
        self.update_paddle_physics()
        events, done, winner = self._update_ball_physics()
        self.last_result = StepResult(self.get_states(), frozenset(events), done, winner)
        return self.last_result

    def update(self):
        result = self._advance()
        return result.events, result.done, result.winner

    def update_paddle_physics(self):
        for player in Player:
            self._integrate_slime(player)

    def _integrate_slime(self, player: Player):
        vertical_velocity = self._get_slime_attribute(player, "vy") + self.state.GRAVITY
        top_y = self._get_slime_attribute(player, "y") + vertical_velocity
        if top_y >= self.state.SLIME_GROUND_TOP_Y:
            top_y = self.state.SLIME_GROUND_TOP_Y
            vertical_velocity = 0.0
        self._set_slime_attribute(player, "y", top_y)
        self._set_slime_attribute(player, "vy", vertical_velocity)

        horizontal_velocity = self._get_slime_attribute(player, "vx")
        min_x, max_x = self._slime_x_bounds(player)
        unclamped_x = self._get_slime_attribute(player, "x") + horizontal_velocity
        left_x = clamp(unclamped_x, min_x, max_x)
        if left_x != unclamped_x:
            self._set_slime_attribute(player, "vx", 0.0)
        self._set_slime_attribute(player, "x", left_x)

    def _ball_side(self) -> Player:
        if self.state.ball_x < self.state.NET_CENTER_X:
            return Player.PLAYER_ONE
        return Player.PLAYER_TWO

    def _update_ball_physics(self):
        events = set()
        previous_side = self._ball_side()

        self.state.ball_vy += self.state.GRAVITY
        self._limit_ball_speed()
        self.state.ball_x += self.state.ball_vx
        self.state.ball_y += self.state.ball_vy

        for player in Player:
            if self._collides_with_paddle(player):
                events.add(HIT_EVENT_BY_PLAYER[player])
        self._bounce_off_net()
        self._bounce_off_walls_and_ceiling()

        current_side = self._ball_side()
        if current_side is not previous_side:
            events.add(CROSSING_EVENT_BY_DESTINATION[current_side])

        if self.state.ball_y < self.state.GROUND_Y - self.state.BALL_RADIUS:
            return events, False, None

        self.state.ball_y = self.state.GROUND_Y - self.state.BALL_RADIUS
        winner = Player.PLAYER_TWO if current_side is Player.PLAYER_ONE else Player.PLAYER_ONE
        self._award_point(winner)
        events.add(ROUND_WIN_EVENT_BY_PLAYER[winner])
        self.reset_paddle_positions()
        self.reset_ball(winner)
        return events, True, winner

    def _award_point(self, winner: Player):
        if winner is Player.PLAYER_ONE:
            self.state.player_score += 1
        else:
            self.state.opponent_score += 1

    def update_ball_physics(self):
        events, done, winner = self._update_ball_physics()
        self.last_result = StepResult(self.get_states(), frozenset(events), done, winner)
        return frozenset(events), done, winner

    def _limit_ball_speed(self):
        speed = math.hypot(self.state.ball_vx, self.state.ball_vy)
        if speed > self.state.BALL_MAX_SPEED:
            scale = self.state.BALL_MAX_SPEED / speed
            self.state.ball_vx *= scale
            self.state.ball_vy *= scale

    def _bounce_off_walls_and_ceiling(self):
        radius = self.state.BALL_RADIUS
        if self.state.ball_x < radius:
            self.state.ball_x = radius
            self.state.ball_vx = abs(self.state.ball_vx)
        elif self.state.ball_x > self.state.SCREEN_WIDTH - radius:
            self.state.ball_x = self.state.SCREEN_WIDTH - radius
            self.state.ball_vx = -abs(self.state.ball_vx)
        if self.state.ball_y < radius:
            self.state.ball_y = radius
            self.state.ball_vy = abs(self.state.ball_vy)

    def _slime_contact(self, center_x: float, center_y: float) -> Optional[Contact]:
        slime_radius = self.state.PADDLE_RADIUS
        ball_radius = self.state.BALL_RADIUS
        ball_is_above_slime_base = self.state.ball_y <= center_y
        if ball_is_above_slime_base:
            anchor_x, anchor_y = center_x, center_y
            contact_reach = slime_radius + ball_radius
        else:
            anchor_x = clamp(self.state.ball_x, center_x - slime_radius, center_x + slime_radius)
            anchor_y = center_y
            contact_reach = ball_radius
        return self._circle_contact(anchor_x, anchor_y, contact_reach, ball_is_above_slime_base)

    def _net_contact(self) -> Optional[Contact]:
        net_left = self.state.NET_X
        net_right = self.state.NET_X + self.state.NET_WIDTH
        net_top = self.state.GROUND_Y - self.state.NET_HEIGHT
        closest_x = clamp(self.state.ball_x, net_left, net_right)
        closest_y = clamp(self.state.ball_y, net_top, self.state.GROUND_Y)
        ball_is_above_net = self.state.ball_y <= net_top
        if closest_x == self.state.ball_x and closest_y == self.state.ball_y:
            return self._contact_from_inside_net(net_left, net_right, net_top)
        return self._circle_contact(closest_x, closest_y, self.state.BALL_RADIUS, ball_is_above_net)

    def _contact_from_inside_net(self, net_left: float, net_right: float, net_top: float) -> Contact:
        radius = self.state.BALL_RADIUS
        exit_distances = {
            (0.0, -1.0): self.state.ball_y - net_top + radius,
            (-1.0, 0.0): self.state.ball_x - net_left + radius,
            (1.0, 0.0): net_right - self.state.ball_x + radius,
        }
        (normal_x, normal_y), penetration = min(exit_distances.items(), key=lambda item: item[1])
        return Contact(normal_x, normal_y, penetration)

    def _circle_contact(
        self, anchor_x: float, anchor_y: float, contact_reach: float, push_upward_when_centered: bool
    ) -> Optional[Contact]:
        offset_x = self.state.ball_x - anchor_x
        offset_y = self.state.ball_y - anchor_y
        distance = math.hypot(offset_x, offset_y)
        if distance >= contact_reach:
            return None
        if distance == 0:
            return Contact(0.0, -1.0 if push_upward_when_centered else 1.0, contact_reach)
        return Contact(offset_x / distance, offset_y / distance, contact_reach - distance)

    def _resolve_ball_contact(self, contact: Contact, surface_vx: float, surface_vy: float) -> bool:
        self.state.ball_x += contact.normal_x * contact.penetration
        self.state.ball_y += contact.normal_y * contact.penetration
        relative_vx = self.state.ball_vx - surface_vx
        relative_vy = self.state.ball_vy - surface_vy
        approach_speed = relative_vx * contact.normal_x + relative_vy * contact.normal_y
        if approach_speed >= 0:
            return False
        rebound_impulse = (1 + self.state.collision_coefficient) * approach_speed
        self.state.ball_vx = relative_vx - rebound_impulse * contact.normal_x + surface_vx
        self.state.ball_vy = relative_vy - rebound_impulse * contact.normal_y + surface_vy
        self._limit_ball_speed()
        return True

    def _collides_with_paddle(self, player: Player) -> bool:
        contact = self._slime_contact(*self._slime_center(player))
        if contact is None:
            return False
        return self._resolve_ball_contact(
            contact,
            self._get_slime_attribute(player, "vx"),
            self._get_slime_attribute(player, "vy"),
        )

    def _bounce_off_net(self) -> bool:
        contact = self._net_contact()
        if contact is None:
            return False
        return self._resolve_ball_contact(contact, 0.0, 0.0)

    def get_state(self, player: Union[Player, str] = Player.PLAYER_ONE) -> List[float]:
        player = self._coerce_player(player)
        values = [
            self.state.ball_x, self.state.ball_y, self.state.ball_vx, self.state.ball_vy,
            self.state.player_x, self.state.player_y, self.state.player_vy,
            self.state.opponent_x, self.state.opponent_y, self.state.opponent_vy,
        ]
        if player is Player.PLAYER_ONE:
            return values

        diameter = 2 * self.state.PADDLE_RADIUS
        return [
            self.state.SCREEN_WIDTH - self.state.ball_x, self.state.ball_y, -self.state.ball_vx, self.state.ball_vy,
            self.state.SCREEN_WIDTH - self.state.opponent_x - diameter, self.state.opponent_y, self.state.opponent_vy,
            self.state.SCREEN_WIDTH - self.state.player_x - diameter, self.state.player_y, self.state.player_vy,
        ]

    def get_states(self) -> Dict[str, List[float]]:
        return {
            Player.PLAYER_ONE.value: self.get_state(Player.PLAYER_ONE),
            Player.PLAYER_TWO.value: self.get_state(Player.PLAYER_TWO),
        }

    @staticmethod
    def _held_direction(keys, left_key, right_key) -> int:
        return (RIGHT_DIRECTION if keys[right_key] else NO_DIRECTION) + (
            LEFT_DIRECTION if keys[left_key] else NO_DIRECTION
        )

    def handle_input(self, control_opponent: bool = True):
        if pygame is None:
            raise RuntimeError("pygame is required for keyboard input")
        keys = pygame.key.get_pressed()
        self._steer_slime(Player.PLAYER_ONE, self._held_direction(keys, pygame.K_a, pygame.K_d))
        if keys[pygame.K_w]:
            self._jump_slime(Player.PLAYER_ONE)
        if control_opponent:
            self._steer_slime(Player.PLAYER_TWO, self._held_direction(keys, pygame.K_LEFT, pygame.K_RIGHT))
            if keys[pygame.K_UP]:
                self._jump_slime(Player.PLAYER_TWO)
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

    def _quit_requested(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def play_against_agent(self, agent):
        if pygame is None:
            raise RuntimeError("pygame is required to play against an agent")
        if self.screen is None:
            self._initialize_renderer()
        agent.model.eval()
        self.running = True
        while self.running:
            self.running = not self._quit_requested()
            self.handle_input(control_opponent=False)
            ai_action = self._agent_action(agent, self.get_state(Player.PLAYER_TWO))
            self._apply_action(Player.PLAYER_TWO, ai_action)
            self.update()
            self.render()
            self.clock.tick(FRAMES_PER_SECOND)
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
        pygame.draw.circle(surface, eye_color, (int(eye_x), int(eye_y)), int(eye_radius), 1)

    def _draw_slime(self, player: Player, color, which_paddle: str):
        center_x, center_y = self._slime_center(player)
        center = (int(center_x), int(center_y))
        radius = int(self.state.PADDLE_RADIUS)
        self.draw_semicircle(self.screen, color, center, radius, True)
        self.draw_paddle_eyes(
            self.screen, center, radius, self.BLACK, self.state.ball_x, self.state.ball_y, which_paddle
        )

    def render(self):
        if pygame is None:
            raise RuntimeError("pygame is required for rendering")
        if self.screen is None:
            self._initialize_renderer()
        self.screen.fill(self.BLACK)
        pygame.draw.rect(
            self.screen,
            self.GREEN,
            (0, self.state.GROUND_Y, self.state.SCREEN_WIDTH, self.state.SCREEN_HEIGHT - self.state.GROUND_Y),
        )
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            (self.state.NET_X, self.state.GROUND_Y - self.state.NET_HEIGHT, self.state.NET_WIDTH, self.state.NET_HEIGHT),
        )
        self._draw_slime(Player.PLAYER_ONE, self.BLUE, "player")
        self._draw_slime(Player.PLAYER_TWO, self.RED, "opponent")
        pygame.draw.circle(
            self.screen, self.WHITE, (int(self.state.ball_x), int(self.state.ball_y)), int(self.state.BALL_RADIUS)
        )
        self.screen.blit(
            self.score_font.render(f"Player: {self.state.player_score}", True, self.BLUE),
            (SCORE_MARGIN_PIXELS, SCORE_MARGIN_PIXELS),
        )
        opponent_score_text = self.score_font.render(f"Opponent: {self.state.opponent_score}", True, self.RED)
        self.screen.blit(
            opponent_score_text,
            (self.state.SCREEN_WIDTH - opponent_score_text.get_width() - SCORE_MARGIN_PIXELS, SCORE_MARGIN_PIXELS),
        )
        pygame.display.flip()

    def run(self, manual_control=True):
        if pygame is None:
            raise RuntimeError("pygame is required to run the manual game")
        if self.screen is None:
            self._initialize_renderer()
        while self.running:
            self.running = not self._quit_requested()
            if manual_control:
                self.handle_input()
            self.update()
            self.render()
            self.clock.tick(FRAMES_PER_SECOND)
        pygame.quit()


if __name__ == "__main__":
    game = SlimeVolleyball(render=True)
    if len(sys.argv) > 1:
        game.play_against_champion(sys.argv[1])
    else:
        game.run()
