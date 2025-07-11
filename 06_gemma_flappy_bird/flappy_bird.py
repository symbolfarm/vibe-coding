import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
width = 600
height = 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flappy Bird")

# Colors
light_colors = [(173, 216, 230), (240, 255, 255), (240, 248, 255), (224, 255, 255)]  # Light shades
dark_colors = [(0, 0, 139), (0, 0, 205), (0, 0, 128), (160, 82, 45), (128, 0, 0)]  # Dark shades
pipe_colors = [(0, 100, 0), (139, 69, 19), (85, 85, 85)] # Dark green, brown, gray
land_colors = [(139, 69, 19), (255, 255, 0)] # Brown, Yellow

# Random initial background color
bg_color = random.choice(light_colors)

# Bird properties
bird_x = 50
bird_y = height // 2
bird_radius = 20
bird_shape = random.choice(["circle", "square", "triangle"])
bird_color = random.choice(dark_colors)
bird_velocity = 0
gravity = 0.5
acceleration = 0.2

# Land properties
land_height = 50
land_color = random.choice(land_colors)

# Pipe properties
pipe_width = 50
pipe_gap = 150
pipe_speed = 2
pipes = []
last_pipe_time = pygame.time.get_ticks()
pipe_frequency = 1500 # Milliseconds between pipes

# Score
score = 0
font = pygame.font.Font(None, 36)
best_score = 0

# Game over flag
game_over = False
game_over_text = font.render("Game Over! Press SPACE to restart, Q or Esc to quit.", True, (255, 255, 255))
game_over_rect = game_over_text.get_rect(center=(width // 2, height // 2))

# Best Score Text
best_score_text = font.render(f"Best Score: {best_score}", True, (255, 255, 255))
best_score_rect = best_score_text.get_rect(topright=(width - 10, 10))

# Function to draw the bird
def draw_bird():
    if bird_shape == "circle":
        pygame.draw.circle(screen, bird_color, (bird_x, bird_y), bird_radius)
    elif bird_shape == "square":
        pygame.draw.rect(screen, bird_color, (bird_x - bird_radius, bird_y - bird_radius, 2 * bird_radius, 2 * bird_radius))
    elif bird_shape == "triangle":
        pygame.draw.polygon(screen, bird_color, [(bird_x, bird_y - bird_radius), (bird_x - bird_radius, bird_y + bird_radius), (bird_x + bird_radius, bird_y + bird_radius)])

# Function to draw the land
def draw_land():
    pygame.draw.rect(screen, land_color, (0, height - land_height, width, land_height))

# Function to draw the pipes
def draw_pipes():
    for pipe in pipes:
        pygame.draw.rect(screen, pipe[2], (pipe[0], 0, pipe_width, pipe[1]))  # Top pipe
        pygame.draw.rect(screen, pipe[2], (pipe[0], pipe[1] + pipe_gap, pipe_width, height - pipe[1] - pipe_gap))  # Bottom pipe

# Function to generate a new pipe
def create_pipe():
    pipe_height = random.randint(50, height - land_height - pipe_gap - 50)
    pipe_color = random.choice(pipe_colors)
    pipes.append([width, pipe_height, pipe_color])

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_over:
                bird_velocity = -10
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False

    if not game_over:
        # Update bird position
        bird_velocity += gravity
        bird_y += bird_velocity
        if bird_y > height - land_height:
            game_over = True

        # Generate pipes
        current_time = pygame.time.get_ticks()
        if current_time - last_pipe_time > pipe_frequency:
            create_pipe()
            last_pipe_time = current_time

        # Move pipes
        for pipe in pipes:
            pipe[0] -= pipe_speed

        # Remove off-screen pipes
        pipes = [pipe for pipe in pipes if pipe[0] > -pipe_width]

        # Collision detection
        for pipe in pipes:
            if (bird_x + bird_radius > pipe[0] and bird_x - bird_radius < pipe[0] + pipe_width and
                (bird_y - bird_radius < pipe[1] or bird_y + bird_radius > pipe[1] + pipe_gap)):
                game_over = True

        # Score increment
        for pipe in pipes:
            if pipe[0] + pipe_width < bird_x and not pipe[3]:  # Only increment once per pipe
                score += 1
                pipe[3] = True

        # Draw everything
        screen.fill(bg_color)
        draw_land()
        draw_pipes()
        draw_bird()

        # Display score
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (width - 100, 10))
        screen.blit(best_score_text, best_score_rect)

    else:
        screen.fill(bg_color)
        screen.blit(game_over_text, game_over_rect)
        best_score = max(score, best_score)
        best_score_text = font.render(f"Best Score: {best_score}", True, (255, 255, 255))
        screen.blit(best_score_text, best_score_rect)


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
