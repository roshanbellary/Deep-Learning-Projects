from dataclasses import dataclass

import numpy as np

OBSERVATION_DIVISOR = 10.0
OWN_X_INDEX = 0
BALL_X_INDEX = 4
BALL_Y_INDEX = 5
BALL_VX_INDEX = 6
BALL_VY_INDEX = 7

TIMESTEP = 1.0 / 30.0
GRAVITY = -9.8 * 2 * 1.5
MAX_BALL_SPEED = 15 * 1.5
GROUND_HEIGHT = 1.5
HALF_COURT_WIDTH = 24.0
BALL_RADIUS = 0.5
SLIME_RADIUS = 1.5
FENCE_HALF_WIDTH = 0.5
FENCE_HEIGHT = 3.5
NEAREST_SLIME_X = FENCE_HALF_WIDTH + SLIME_RADIUS
FARTHEST_SLIME_X = HALF_COURT_WIDTH - SLIME_RADIUS
WALL_X = HALF_COURT_WIDTH - BALL_RADIUS
FENCE_CONTACT_X = FENCE_HALF_WIDTH + BALL_RADIUS

PREDICTION_HORIZON_STEPS = 150

AWAY_FROM_NET_ACTION = 0
TOWARD_NET_ACTION = 1
JUMP_ACTION = 2
STAY_ACTION = 3


@dataclass(frozen=True)
class BotSettings:
    contact_height: float = GROUND_HEIGHT + SLIME_RADIUS
    return_offset: float = 0.3
    position_tolerance: float = 0.3
    home_x: float = 12.0
    jump_reach: float = 2.0
    jump_height: float = 9.0


DEFAULT_SETTINGS = BotSettings()


def predict_landing_x(ball_x, ball_y, ball_vx, ball_vy, contact_height: float) -> np.ndarray:
    x = ball_x.astype(np.float64).copy()
    y = ball_y.astype(np.float64).copy()
    vx = ball_vx.astype(np.float64).copy()
    vy = ball_vy.astype(np.float64).copy()
    landing_x = np.full_like(x, np.nan)
    still_flying = np.ones_like(x, dtype=bool)
    for _ in range(PREDICTION_HORIZON_STEPS):
        vy = vy + GRAVITY * TIMESTEP
        speed = np.hypot(vx, vy)
        over_limit = speed > MAX_BALL_SPEED
        speed_scale = np.where(over_limit, MAX_BALL_SPEED / np.maximum(speed, 1e-9), 1.0)
        vx = vx * speed_scale
        vy = vy * speed_scale
        previous_x = x
        x = x + vx * TIMESTEP
        y = y + vy * TIMESTEP
        hit_wall = np.abs(x) >= WALL_X
        vx = np.where(hit_wall, -vx, vx)
        x = np.clip(x, -WALL_X, WALL_X)
        hit_fence = (np.abs(x) <= FENCE_CONTACT_X) & (np.abs(previous_x) > FENCE_CONTACT_X) & (y <= FENCE_HEIGHT)
        vx = np.where(hit_fence, -vx, vx)
        x = np.where(hit_fence, np.sign(previous_x) * FENCE_CONTACT_X, x)
        newly_landed = still_flying & (y <= contact_height) & (vy < 0)
        landing_x = np.where(newly_landed, x, landing_x)
        still_flying &= ~newly_landed
        if not still_flying.any():
            break
    return landing_x


def choose_actions(observations: np.ndarray, settings: BotSettings = DEFAULT_SETTINGS) -> np.ndarray:
    world_values = observations * OBSERVATION_DIVISOR
    own_x = world_values[:, OWN_X_INDEX]
    ball_x = world_values[:, BALL_X_INDEX]
    ball_y = world_values[:, BALL_Y_INDEX]
    ball_vx = world_values[:, BALL_VX_INDEX]
    ball_vy = world_values[:, BALL_VY_INDEX]

    ball_heading_toward_bot = (own_x - ball_x) * ball_vx > 0
    landing_x = predict_landing_x(ball_x, ball_y, ball_vx, ball_vy, settings.contact_height)
    lands_on_bot_side = np.nan_to_num(landing_x, nan=-1.0) > 0
    should_intercept = ball_heading_toward_bot & lands_on_bot_side

    intercept_x = np.nan_to_num(landing_x, nan=settings.home_x) + settings.return_offset
    target_x = np.clip(np.where(should_intercept, intercept_x, settings.home_x), NEAREST_SLIME_X, FARTHEST_SLIME_X)

    actions = np.full(len(observations), STAY_ACTION)
    actions[own_x < target_x - settings.position_tolerance] = AWAY_FROM_NET_ACTION
    actions[own_x > target_x + settings.position_tolerance] = TOWARD_NET_ACTION

    ball_is_close = np.abs(ball_x - own_x) < settings.jump_reach
    ball_is_low_and_falling = (ball_y < settings.jump_height) & (ball_vy < 0)
    actions[should_intercept & ball_is_close & ball_is_low_and_falling] = JUMP_ACTION
    return actions
