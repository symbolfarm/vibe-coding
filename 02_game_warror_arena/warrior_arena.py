import pygame
import sys
import random
import math
from pygame.locals import *

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 32
PLAYER_SPEED = 3
ENEMY_SPEED = 1
SWORD_COOLDOWN = 20
PLAYER_START_HEALTH = 5
FONT_SIZE = 24

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Warror Arena')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, FONT_SIZE)

class Player:
    def __init__(self):
        self.width = TILE_SIZE
        self.height = TILE_SIZE
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed = PLAYER_SPEED
        self.facing = 'down'
        self.health = PLAYER_START_HEALTH
        self.score = 0
        self.sword_active = False
        self.sword_cooldown = 0
        
    def update(self, keys):
        # Movement
        dx, dy = 0, 0
        if keys[K_LEFT] or keys[K_a]:
            dx = -self.speed
            self.facing = 'left'
        if keys[K_RIGHT] or keys[K_d]:
            dx = self.speed
            self.facing = 'right'
        if keys[K_UP] or keys[K_w]:
            dy = -self.speed
            self.facing = 'up'
        if keys[K_DOWN] or keys[K_s]:
            dy = self.speed
            self.facing = 'down'
            
        # Stay within screen bounds
        self.x = max(0, min(self.x + dx, SCREEN_WIDTH - self.width))
        self.y = max(0, min(self.y + dy, SCREEN_HEIGHT - self.height))
        
        # Sword attack
        if self.sword_cooldown > 0:
            self.sword_cooldown -= 1
            
        if (keys[K_SPACE] or keys[K_j]) and self.sword_cooldown == 0:
            self.sword_active = True
            self.sword_cooldown = SWORD_COOLDOWN
        else:
            self.sword_active = False
            
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def get_sword_rect(self):
        sword_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        if self.facing == 'up':
            sword_rect.y -= self.height
        elif self.facing == 'down':
            sword_rect.y += self.height
        elif self.facing == 'left':
            sword_rect.x -= self.width
        elif self.facing == 'right':
            sword_rect.x += self.width
        return sword_rect
    
    def draw(self, screen):
        # Draw player
        pygame.draw.rect(screen, BLUE, self.get_rect())
        
        # Draw sword if active
        if self.sword_active:
            pygame.draw.rect(screen, YELLOW, self.get_sword_rect())
            
        # Draw health
        for i in range(self.health):
            pygame.draw.rect(screen, RED, (10 + i * 20, 10, 15, 15))

class Enemy:
    def __init__(self):
        self.width = TILE_SIZE
        self.height = TILE_SIZE
        self.speed = ENEMY_SPEED
        
        # Spawn enemy at a random edge of the screen
        side = random.randint(0, 3)
        if side == 0:  # Top
            self.x = random.randint(0, SCREEN_WIDTH - self.width)
            self.y = -self.height
        elif side == 1:  # Right
            self.x = SCREEN_WIDTH
            self.y = random.randint(0, SCREEN_HEIGHT - self.height)
        elif side == 2:  # Bottom
            self.x = random.randint(0, SCREEN_WIDTH - self.width)
            self.y = SCREEN_HEIGHT
        else:  # Left
            self.x = -self.width
            self.y = random.randint(0, SCREEN_HEIGHT - self.height)
            
        self.move_timer = random.randint(30, 90)
        self.direction = random.choice(['up', 'down', 'left', 'right'])
        
    def update(self, player):
        # Move towards player with some randomness
        self.move_timer -= 1
        if self.move_timer <= 0:
            self.move_timer = random.randint(30, 90)
            self.direction = random.choice(['up', 'down', 'left', 'right'])
            
            # 70% chance to move towards player
            if random.random() < 0.7:
                if player.x < self.x:
                    self.direction = 'left'
                elif player.x > self.x:
                    self.direction = 'right'
                elif player.y < self.y:
                    self.direction = 'up'
                elif player.y > self.y:
                    self.direction = 'down'
        
        # Move in current direction
        if self.direction == 'up':
            self.y -= self.speed
        elif self.direction == 'down':
            self.y += self.speed
        elif self.direction == 'left':
            self.x -= self.speed
        elif self.direction == 'right':
            self.x += self.speed
            
        # Keep enemies inside the screen
        if self.x < 0:
            self.x = 0
            self.direction = 'right'
        elif self.x > SCREEN_WIDTH - self.width:
            self.x = SCREEN_WIDTH - self.width
            self.direction = 'left'
        if self.y < 0:
            self.y = 0
            self.direction = 'down'
        elif self.y > SCREEN_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - self.height
            self.direction = 'up'
            
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.get_rect())

def draw_arena():
    # Draw checkerboard pattern for the arena
    for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
        for x in range(0, SCREEN_WIDTH, TILE_SIZE):
            if (x // TILE_SIZE + y // TILE_SIZE) % 2 == 0:
                pygame.draw.rect(screen, GREEN, (x, y, TILE_SIZE, TILE_SIZE))
            else:
                pygame.draw.rect(screen, DARK_GREEN, (x, y, TILE_SIZE, TILE_SIZE))

def show_game_over(screen, score):
    screen.fill(BLACK)
    title_text = font.render(f'GAME OVER', True, WHITE)
    score_text = font.render(f'Final Score: {score}', True, WHITE)
    restart_text = font.render('Press R to restart or Q to quit', True, WHITE)
    
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
    
    screen.blit(title_text, title_rect)
    screen.blit(score_text, score_rect)
    screen.blit(restart_text, restart_rect)
    pygame.display.flip()

def main():
    player = Player()
    enemies = []
    spawn_timer = 0
    game_over = False
    
    # Main game loop
    while True:
        if game_over:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_r:
                        # Restart game
                        return main()
                    elif event.key == K_q:
                        pygame.quit()
                        sys.exit()
            
            show_game_over(screen, player.score)
            clock.tick(60)
            continue
            
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Update player
        player.update(keys)
        
        # Spawn enemies
        spawn_timer -= 1
        if spawn_timer <= 0:
            enemies.append(Enemy())
            spawn_timer = 120  # Spawn a new enemy every 2 seconds
        
        # Update enemies
        for enemy in enemies[:]:
            enemy.update(player)
            
            # Check for sword hits
            if player.sword_active and enemy.get_rect().colliderect(player.get_sword_rect()):
                enemies.remove(enemy)
                player.score += 10
                continue
                
            # Check for collision with player
            if enemy.get_rect().colliderect(player.get_rect()):
                player.health -= 1
                enemies.remove(enemy)
                if player.health <= 0:
                    game_over = True
        
        # Draw everything
        draw_arena()
        player.draw(screen)
        for enemy in enemies:
            enemy.draw(screen)
            
        # Draw score
        score_text = font.render(f'Score: {player.score}', True, WHITE)
        screen.blit(score_text, (SCREEN_WIDTH - 150, 10))
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
