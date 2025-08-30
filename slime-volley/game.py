import pygame
from pygame.locals import *
import sys
import math
from agent import *

class GameState:
    """Contains all game state data"""
    def __init__(self):
        # Screen dimensions
        self.SCREEN_WIDTH = 750
        self.SCREEN_HEIGHT = 370
        self.GROUND_Y = 300
        # Game constants
        self.PADDLE_RADIUS = 40

        # Player and opponent positions
        self.player_x = 125
        self.player_y = self.GROUND_Y - self.PADDLE_RADIUS
        self.player_vy = 0
        self.opponent_x = 562.5
        self.opponent_y = self.GROUND_Y - self.PADDLE_RADIUS
        self.opponent_vy = 0
        
        # Ball position and velocity
        self.ball_x = 375
        self.ball_y = 100
        self.ball_vx = 0
        self.ball_vy = 0
        
        self.MOVEMENT_SPEED = 2

        self.BALL_RADIUS = 10        
        self.BALL_SPEED = 4
        self.BALL_MAX_SPEED = 12
        
        self.GRAVITY = 0.4
        self.JUMP_FORCE = -6  # Negative for upward movement
        
        self.collision_coefficient = 1
        self.collision_margin = 0
        # Scores
        self.player_score = 0
        self.opponent_score = 0
        
        # Game boundaries
        self.NET_X = 375
        self.NET_WIDTH = 5
        self.NET_HEIGHT = 50

class SlimeVolleyball:
    """Main game controller class"""
    def __init__(self):
        pygame.init()
        self.state = GameState()
        self.screen = pygame.display.set_mode((self.state.SCREEN_WIDTH, self.state.SCREEN_HEIGHT))
        pygame.display.set_caption("Slime Volleyball")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)
        self.GRAY = (128, 128, 128)
        
        self.reset_ball()
    
    def reset_ball(self, winner="player"):
        """Reset ball over the winning player's paddle with upward movement only"""
        if winner == "player":
            self.state.ball_x = self.state.player_x + self.state.PADDLE_RADIUS
            self.state.ball_y = self.state.player_y - self.state.BALL_RADIUS - 10
            self.state.ball_vx = 0
            self.state.ball_vy = -self.state.BALL_SPEED
        else:
            self.state.ball_x = self.state.opponent_x + self.state.PADDLE_RADIUS
            self.state.ball_y = self.state.opponent_y - self.state.BALL_RADIUS - 10
            self.state.ball_vx = 0
            self.state.ball_vy = -self.state.BALL_SPEED
    
    def move_player(self, direction):
        """Move player sprite based on direction (-1 for left, 1 for right)"""
        new_x = self.state.player_x + (direction * self.state.MOVEMENT_SPEED)
        # Keep player within left half of screen (accounting for semicircular paddle)
        if 0 <= new_x <= self.state.NET_X - self.state.PADDLE_RADIUS * 2:
            self.state.player_x = new_x
    
    def jump_player(self):
        """Make player jump if on ground"""
        if self.state.player_y >= self.state.GROUND_Y - self.state.PADDLE_RADIUS:
            self.state.player_vy = self.state.JUMP_FORCE
    
    def move_opponent(self, direction):
        """Move opponent sprite based on direction (-1 for left, 1 for right)"""
        new_x = self.state.opponent_x + (direction * self.state.MOVEMENT_SPEED)
        # Keep opponent within right half of screen (accounting for semicircular paddle)
        if self.state.NET_X + self.state.NET_WIDTH <= new_x <= self.state.SCREEN_WIDTH - self.state.PADDLE_RADIUS * 2:
            self.state.opponent_x = new_x
    
    def jump_opponent(self):
        """Make opponent jump if on ground"""
        if self.state.opponent_y >= self.state.GROUND_Y - self.state.PADDLE_RADIUS:
            self.state.opponent_vy = self.state.JUMP_FORCE
    
    def calculate_ball_velocity_after_collision(self, paddle_x, paddle_y, ball_x, ball_y, ball_vx, ball_vy, paddle_vx, paddle_vy):
        """Calculate paddle-ball collision and return new ball velocity"""
        distance_vector = (paddle_x - ball_x, paddle_y - ball_y)

        v_ref = (ball_vx - paddle_vx, ball_vy - paddle_vy)

        dot_product = distance_vector[0] * v_ref[0] + distance_vector[1] * v_ref[1]

        v_ball_new = (v_ref[0] - 2 *(dot_product / (distance_vector[0]**2 + distance_vector[1]**2)) * distance_vector[0],
                      v_ref[1] - 2 *(dot_product / (distance_vector[0]**2 + distance_vector[1]**2)) * distance_vector[1])

        v_ball_new = (v_ball_new[0] + paddle_vx, v_ball_new[1] + paddle_vy)
        
        v_ball_new = (v_ball_new[0] * self.state.collision_coefficient, v_ball_new[1] * self.state.collision_coefficient)
        
        v_ball_mag = math.sqrt(v_ball_new[0]**2 + v_ball_new[1]**2)
        if v_ball_mag > self.state.BALL_MAX_SPEED:
            v_ball_new = (v_ball_new[0] * self.state.BALL_MAX_SPEED / v_ball_mag, v_ball_new[1] * self.state.BALL_MAX_SPEED / v_ball_mag)
        
        return v_ball_new 
    

    def update_paddle_physics(self):
        """Update paddle physics including gravity and ground collision"""
        # Update player physics
        self.state.player_y += self.state.player_vy
        self.state.player_vy += self.state.GRAVITY
        
        # Ground collision for player
        if self.state.player_y > self.state.GROUND_Y - self.state.PADDLE_RADIUS:
            self.state.player_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
            self.state.player_vy = 0
        
        # Update opponent physics
        self.state.opponent_y += self.state.opponent_vy
        self.state.opponent_vy += self.state.GRAVITY
        
        # Ground collision for opponent
        if self.state.opponent_y > self.state.GROUND_Y - self.state.PADDLE_RADIUS:
            self.state.opponent_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
            self.state.opponent_vy = 0
    
    def update_ball_physics(self):
        """Update ball physics including movement, gravity, and collisions"""
        player_reward = 0
        opponent_reward = 0
        done = False
        # Update ball position
        self.state.ball_x += self.state.ball_vx
        self.state.ball_y += self.state.ball_vy
        
        # Apply gravity
        self.state.ball_vy += self.state.GRAVITY
        
        # Ball collision with left and right walls
        if self.state.ball_x <= self.state.BALL_RADIUS or self.state.ball_x >= self.state.SCREEN_WIDTH - self.state.BALL_RADIUS:
            self.state.ball_vx *= -1
            self.state.ball_x = max(self.state.BALL_RADIUS, min(self.state.SCREEN_WIDTH - self.state.BALL_RADIUS, self.state.ball_x))
        
        # Ball collision with ceiling
        if self.state.ball_y <= self.state.BALL_RADIUS:
            self.state.ball_vy *= -self.state.collision_coefficient  # Slight energy loss
            self.state.ball_y = self.state.BALL_RADIUS
        
        # Ball collision with ground
        if self.state.ball_y >= self.state.GROUND_Y - self.state.BALL_RADIUS:
            # Determine which side the ball landed on
            if self.state.ball_x < self.state.NET_X:
                self.state.opponent_score += 1
                self.reset_paddle_positions()
                self.reset_ball("opponent")
                player_reward -= 100
                opponent_reward += 100
                done = True
                return (player_reward, opponent_reward, done)
            else:
                self.state.player_score += 1
                self.reset_paddle_positions()
                self.reset_ball("player")
                player_reward += 100
                opponent_reward -= 100
                done = True 
                return (player_reward, opponent_reward, done)
        
        if self.state.ball_x < self.state.NET_X:
            player_reward -= 1
        else:
            opponent_reward -= 1

        # Ball collision with net
        if (self.state.NET_X - self.state.BALL_RADIUS <= self.state.ball_x <= self.state.NET_X + self.state.NET_WIDTH + self.state.BALL_RADIUS and
            self.state.ball_y >= self.state.GROUND_Y - self.state.NET_HEIGHT - self.state.BALL_RADIUS):
            self.state.ball_vx *= -1
        
        # Ball collision with player paddle (semicircle)
        paddle_center_x = self.state.player_x + self.state.PADDLE_RADIUS
        paddle_center_y = self.state.player_y + self.state.PADDLE_RADIUS
        distance = math.sqrt((self.state.ball_x - paddle_center_x)**2 + (self.state.ball_y - paddle_center_y)**2)
        
        if (distance <= (self.state.PADDLE_RADIUS + self.state.BALL_RADIUS) * (1 + self.state.collision_margin) and self.state.ball_y <= self.state.player_y + self.state.PADDLE_RADIUS):
            # Reposition ball exactly at collision boundary
            collision_distance = self.state.PADDLE_RADIUS + self.state.BALL_RADIUS
            if distance > 0:  # Avoid division by zero
                # Normalize the distance vector and set ball position
                normalized_x = (self.state.ball_x - paddle_center_x) / distance
                normalized_y = (self.state.ball_y - paddle_center_y) / distance
                self.state.ball_x = paddle_center_x + normalized_x * collision_distance
                self.state.ball_y = paddle_center_y + normalized_y * collision_distance
            
            self.state.ball_vx, self.state.ball_vy = self.calculate_ball_velocity_after_collision(paddle_center_x, paddle_center_y, self.state.ball_x, self.state.ball_y, self.state.ball_vx, self.state.ball_vy, 0, self.state.player_vy)
            
            player_reward += 2
        # Ball collision with opponent paddle (semicircle)
        paddle_center_x = self.state.opponent_x + self.state.PADDLE_RADIUS
        paddle_center_y = self.state.opponent_y + self.state.PADDLE_RADIUS
        distance = math.sqrt((self.state.ball_x - paddle_center_x)**2 + (self.state.ball_y - paddle_center_y)**2)
        
        if (distance <= (self.state.PADDLE_RADIUS + self.state.BALL_RADIUS) * (1 + self.state.collision_margin) and self.state.ball_y <= self.state.opponent_y + self.state.PADDLE_RADIUS):
            # Reposition ball exactly at collision boundary
            collision_distance = self.state.PADDLE_RADIUS + self.state.BALL_RADIUS
            if distance > 0:  # Avoid division by zero
                # Normalize the distance vector and set ball position
                normalized_x = (self.state.ball_x - paddle_center_x) / distance
                normalized_y = (self.state.ball_y - paddle_center_y) / distance
                self.state.ball_x = paddle_center_x + normalized_x * collision_distance
                self.state.ball_y = paddle_center_y + normalized_y * collision_distance
            
            self.state.ball_vx, self.state.ball_vy = self.calculate_ball_velocity_after_collision(paddle_center_x, paddle_center_y, self.state.ball_x, self.state.ball_y, self.state.ball_vx, self.state.ball_vy, 0, self.state.opponent_vy)
            opponent_reward += 2

        return (player_reward, opponent_reward, done)
    def handle_input(self):
        """Handle pygame key events and call appropriate movement functions"""
        keys = pygame.key.get_pressed()
        
        # Player movement
        if keys[K_a]:
            self.move_player(-self.state.MOVEMENT_SPEED)
        if keys[K_d]:
            self.move_player(self.state.MOVEMENT_SPEED)
        if keys[K_w] or keys[K_SPACE]:
            self.jump_player()
        
        # Opponent movement (for testing - can be replaced with AI)
        if keys[K_LEFT]:
            self.move_opponent(-self.state.MOVEMENT_SPEED)
        if keys[K_RIGHT]:
            self.move_opponent(self.state.MOVEMENT_SPEED)
        if keys[K_UP]:
            self.jump_opponent()
        
        # Reset ball with R key
        if keys[K_r]:
            self.reset_ball("player")
    
    def draw_semicircle(self, surface, color, center, radius, top_half=True):
        """Draw a filled semicircle using surface operations"""
        # Create a temporary surface for the full circle
        temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        
        # Draw the full circle on the temporary surface
        pygame.draw.circle(temp_surface, color, (radius, radius), radius)
        
        # Create a mask for the semicircle
        if top_half:
            # For top half, clear the bottom half
            pygame.draw.rect(temp_surface, (0, 0, 0, 0), (0, radius, radius * 2, radius))
        else:
            # For bottom half, clear the top half
            pygame.draw.rect(temp_surface, (0, 0, 0, 0), (0, 0, radius * 2, radius))
        
        # Blit the semicircle to the main surface
        surface.blit(temp_surface, (center[0] - radius, center[1] - radius))
    
    def draw_paddle_eyes(self, surface, paddle_center, paddle_radius, eye_color, ball_x, ball_y, which_paddle):
        """Draw eyes on a paddle that always point toward the ball"""
        # Calculate angle between paddle center and ball
        dx = ball_x - paddle_center[0]
        dy = ball_y - paddle_center[1]
        angle = math.atan2(dy, dx)
        
        # Eye positions (slightly above and to the sides of paddle center)
        eye_offset = paddle_radius * 0.3
        left_eye_x = paddle_center[0] - eye_offset
        right_eye_x = paddle_center[0] + eye_offset
        eye_y = paddle_center[1] - eye_offset * 1.5
        
        # Draw eye whites (circles)
        eye_radius = paddle_radius * 0.2
        if which_paddle == "player":
            pygame.draw.circle(surface, self.WHITE, (int(right_eye_x), int(eye_y)), int(eye_radius))
        else:
            pygame.draw.circle(surface, self.WHITE, (int(left_eye_x), int(eye_y)), int(eye_radius))
        
        # Calculate pupil positions based on ball direction
        pupil_offset = eye_radius * 0.4

        if which_paddle == "player":
            right_pupil_x = right_eye_x + math.cos(angle) * pupil_offset
            right_pupil_y = eye_y + math.sin(angle) * pupil_offset
        
        if which_paddle == "opponent":
            left_pupil_x = left_eye_x + math.cos(angle) * pupil_offset
            left_pupil_y = eye_y + math.sin(angle) * pupil_offset
        
        # Draw pupils (black circles)
        pupil_radius = eye_radius * 0.6
        if which_paddle == "player":
            pygame.draw.circle(surface, eye_color, (int(right_pupil_x), int(right_pupil_y)), int(pupil_radius))
        else:
            pygame.draw.circle(surface, eye_color, (int(left_pupil_x), int(left_pupil_y)), int(pupil_radius))
        
        # Draw eye outlines
        if which_paddle == "player":
            pygame.draw.circle(surface, eye_color, (int(right_eye_x), int(eye_y)), int(eye_radius), 2)
        else:
            pygame.draw.circle(surface, eye_color, (int(left_eye_x), int(eye_y)), int(eye_radius), 2)
    def update(self):
        """Update game logic"""
        self.update_paddle_physics()
        return self.update_ball_physics()
    
    def render(self):
        """Render all game elements"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw ground
        pygame.draw.rect(self.screen, self.GREEN, (0, self.state.GROUND_Y, self.state.SCREEN_WIDTH, self.state.SCREEN_HEIGHT - self.state.GROUND_Y))
        
        # Draw net
        pygame.draw.rect(self.screen, self.WHITE, (self.state.NET_X, self.state.GROUND_Y - self.state.NET_HEIGHT, self.state.NET_WIDTH, self.state.NET_HEIGHT))
        
        # Draw player paddle (semicircle) using surface masking
        self.draw_semicircle(self.screen, self.BLUE, (int(self.state.player_x + self.state.PADDLE_RADIUS), int(self.state.player_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, True)
        
        # Draw opponent paddle (semicircle) using surface masking
        self.draw_semicircle(self.screen, self.RED, (int(self.state.opponent_x + self.state.PADDLE_RADIUS), int(self.state.opponent_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, True)
        
        # Draw eyes on both paddles that track the ball
        self.draw_paddle_eyes(self.screen, (int(self.state.player_x + self.state.PADDLE_RADIUS), int(self.state.player_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, self.BLACK, self.state.ball_x, self.state.ball_y, "player")
        self.draw_paddle_eyes(self.screen, (int(self.state.opponent_x + self.state.PADDLE_RADIUS), int(self.state.opponent_y + self.state.PADDLE_RADIUS)), self.state.PADDLE_RADIUS, self.BLACK, self.state.ball_x, self.state.ball_y, "opponent")
        
        # Draw ball
        pygame.draw.circle(self.screen, self.WHITE, (int(self.state.ball_x), int(self.state.ball_y)), self.state.BALL_RADIUS)
        
        # Draw scores
        font = pygame.font.Font(None, 36)
        player_score_text = font.render(f"Player: {self.state.player_score}", True, self.BLUE)
        opponent_score_text = font.render(f"Opponent: {self.state.opponent_score}", True, self.RED)
        self.screen.blit(player_score_text, (10, 10))
        self.screen.blit(opponent_score_text, (self.state.SCREEN_WIDTH - 150, 10))
        
        pygame.display.flip()
    
    def reset_paddle_positions(self):
        """Reset paddle positions to starting positions"""
        self.state.player_x = self.state.SCREEN_WIDTH / 4 - self.state.PADDLE_RADIUS
        self.state.player_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
        self.state.player_vy = 0
        self.state.opponent_x = self.state.SCREEN_WIDTH * 3/4 - self.state.PADDLE_RADIUS
        self.state.opponent_y = self.state.GROUND_Y - self.state.PADDLE_RADIUS
        self.state.opponent_vy = 0
    
    def reset_game(self):
        """Reset game state"""
        self.state = GameState()
        self.reset_ball("player")
    
    def _play_player_action(self, player_action):
        if player_action == 0:
            self.move_player(-1)
        elif player_action == 1:
            self.move_player(1)
        elif player_action == 2:
            self.jump_player()
        elif player_action == 3:
            pass

    def _play_opponent_action(self, opponent_action):
        if opponent_action == 0:
            self.move_opponent(-1)
        elif opponent_action == 1:
            self.move_opponent(1)
        elif opponent_action == 2:
            self.jump_opponent()
        elif opponent_action == 3:
            pass

    def play_step(self, player_action, opponent_action):
        player_action = player_action.index(1)
        opponent_action = opponent_action.index(1)
        self._play_player_action(player_action)
        self._play_opponent_action(opponent_action)
        player_reward, opponent_reward, done = self.update()
        
        # self.render()
        # self.clock.tick(60)
        return player_reward, opponent_reward, done

    def get_state(self):
        return [
            self.state.ball_x,
            self.state.ball_y,
            self.state.ball_vx,
            self.state.ball_vy,
            self.state.player_x,
            self.state.player_y,
            self.state.player_vy,
            self.state.opponent_x,
            self.state.opponent_y,
            self.state.opponent_vy
        ]
        # return {
        #     "ball_x" : self.state.ball_x, 
        #     "ball_y" : self.state.ball_y,
        #     "ball_vx" : self.state.ball_vx,
        #     "ball_vy" : self.state.ball_vy,
        #     "player_x" : self.state.player_x,
        #     "player_y" : self.state.player_y,
        #     "player_vy" : self.state.player_vy,
        #     "opponent_x" : self.state.opponent_x,
        #     "opponent_y" : self.state.opponent_y,
        #     "opponent_vy" : self.state.opponent_vy
        # }

    def play_model(self, type):
        self.agent = Agent(type)
        self.agent.load_model()
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False 
                elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            self.running = False 
            self.handle_input()
            
            action = self.agent._get_best_action(self.get_state()).index(1)
            if type == "opponent":
                print("Opponent AI Action:", action)
                self._play_opponent_action(action)
            else:
                print("Player AI Action:", action)
                self._play_player_action(action)

            self.update()
            self.render()

            self.clock.tick(60)
        pygame.quit()
        sys.exit()
            


    def run(self, manual_control=True):
        """Main game loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
            
            self.handle_input()

            # Update game logic
            self.update()
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = SlimeVolleyball()
    if len(sys.argv) > 1:
        type = sys.argv[1]
        if type.find("player") > 0:
            game.play_model("player")
        else:
            game.play_model("opponent")
    else:
        game.run()

