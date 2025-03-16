import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Shooter")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Player settings
player_size = 50
player_x = WIDTH // 2 - player_size // 2
player_y = HEIGHT - player_size - 20
player_speed = 7

# Enemy settings
enemy_size = 40
enemy_speed = 3
enemies = []
enemy_spawn_rate = 30  # frames between enemy spawns

# Bullet settings
bullet_width = 5
bullet_height = 15
bullet_speed = 10
bullets = []

# Game state
score = 0
game_over = False
clock = pygame.time.Clock()
frame_count = 0

# Load font
font = pygame.font.SysFont(None, 36)

def draw_player(x, y):
    """Draw the player spaceship"""
    # Draw a triangle for the ship
    points = [(x + player_size // 2, y), 
              (x, y + player_size),
              (x + player_size, y + player_size)]
    pygame.draw.polygon(screen, GREEN, points)
    # Draw a small cockpit
    pygame.draw.rect(screen, BLUE, 
                    (x + player_size // 3, y + player_size // 2, 
                     player_size // 3, player_size // 3))

def create_enemy():
    """Create a new enemy at a random position at the top of the screen"""
    x = random.randint(0, WIDTH - enemy_size)
    enemies.append({'x': x, 'y': -enemy_size})

def draw_enemy(enemy):
    """Draw an enemy spaceship"""
    # Draw the enemy body
    pygame.draw.rect(screen, RED, 
                    (enemy['x'], enemy['y'], enemy_size, enemy_size))
    # Draw some details on the enemy ship
    pygame.draw.rect(screen, WHITE, 
                    (enemy['x'] + 10, enemy['y'] + 15, 
                     enemy_size - 20, enemy_size // 4))

def create_bullet(x, y):
    """Create a new bullet at the given position"""
    bullet_x = x + player_size // 2 - bullet_width // 2
    bullets.append({'x': bullet_x, 'y': y})

def draw_bullet(bullet):
    """Draw a bullet"""
    pygame.draw.rect(screen, WHITE, 
                    (bullet['x'], bullet['y'], bullet_width, bullet_height))

def check_collision(bullet, enemy):
    """Check if a bullet collides with an enemy"""
    if (bullet['x'] < enemy['x'] + enemy_size and
        bullet['x'] + bullet_width > enemy['x'] and
        bullet['y'] < enemy['y'] + enemy_size and
        bullet['y'] + bullet_height > enemy['y']):
        return True
    return False

def check_player_collision(player_x, player_y, enemy):
    """Check if the player collides with an enemy"""
    # Use a simplified rectangular collision for the player
    if (player_x < enemy['x'] + enemy_size and
        player_x + player_size > enemy['x'] and
        player_y < enemy['y'] + enemy_size and
        player_y + player_size > enemy['y']):
        return True
    return False

def show_game_over():
    """Display game over screen"""
    game_over_text = font.render("GAME OVER", True, RED)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    restart_text = font.render("Press R to restart or Q to quit", True, WHITE)
    
    screen.blit(game_over_text, 
               (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 50))
    screen.blit(score_text, 
               (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
    screen.blit(restart_text, 
               (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 50))

def reset_game():
    """Reset the game to initial state"""
    global player_x, player_y, enemies, bullets, score, game_over, frame_count
    player_x = WIDTH // 2 - player_size // 2
    player_y = HEIGHT - player_size - 20
    enemies = []
    bullets = []
    score = 0
    game_over = False
    frame_count = 0

# Main game loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_over:
                create_bullet(player_x, player_y)
            if event.key == pygame.K_r and game_over:
                reset_game()
            if event.key == pygame.K_q and game_over:
                pygame.quit()
                sys.exit()
    
    # Get keyboard state for movement
    keys = pygame.key.get_pressed()
    
    if not game_over:
        # Player movement
        if keys[pygame.K_LEFT] and player_x > 0:
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
            player_x += player_speed
        
        # Enemy spawning
        frame_count += 1
        if frame_count % enemy_spawn_rate == 0:
            create_enemy()
        
        # Update enemy positions
        for enemy in enemies[:]:
            enemy['y'] += enemy_speed
            
            # Remove enemies that go off-screen
            if enemy['y'] > HEIGHT:
                enemies.remove(enemy)
            
            # Check for collisions with player
            if check_player_collision(player_x, player_y, enemy):
                game_over = True
        
        # Update bullet positions and check for collisions
        for bullet in bullets[:]:
            bullet['y'] -= bullet_speed
            
            # Remove bullets that go off-screen
            if bullet['y'] < 0:
                bullets.remove(bullet)
                continue
            
            # Check for collisions with enemies
            for enemy in enemies[:]:
                if check_collision(bullet, enemy):
                    if bullet in bullets:
                        bullets.remove(bullet)
                    if enemy in enemies:
                        enemies.remove(enemy)
                    score += 10
                    break
    
    # Drawing
    screen.fill(BLACK)
    
    # Draw stars (background)
    for _ in range(100):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        pygame.draw.circle(screen, WHITE, (x, y), 1)
    
    if not game_over:
        # Draw player
        draw_player(player_x, player_y)
        
        # Draw enemies
        for enemy in enemies:
            draw_enemy(enemy)
        
        # Draw bullets
        for bullet in bullets:
            draw_bullet(bullet)
        
        # Draw score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
    else:
        show_game_over()
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)