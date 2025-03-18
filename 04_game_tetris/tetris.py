import pygame
import random
import sys
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SIDEBAR_WIDTH = 200

# Calculate board position to center it
BOARD_WIDTH = GRID_SIZE * GRID_WIDTH
BOARD_HEIGHT = GRID_SIZE * GRID_HEIGHT
BOARD_X = (SCREEN_WIDTH - BOARD_WIDTH - SIDEBAR_WIDTH) // 2
BOARD_Y = (SCREEN_HEIGHT - BOARD_HEIGHT) // 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Tetromino shapes and colors
SHAPES = [
    [[1, 1, 1, 1]],                                      # I
    [[1, 1], [1, 1]],                                   # O
    [[0, 1, 0], [1, 1, 1]],                             # T
    [[0, 1, 1], [1, 1, 0]],                             # S
    [[1, 1, 0], [0, 1, 1]],                             # Z
    [[1, 0, 0], [1, 1, 1]],                             # J
    [[0, 0, 1], [1, 1, 1]]                              # L
]

COLORS = [
    CYAN,       # I
    YELLOW,     # O
    MAGENTA,    # T
    GREEN,      # S
    RED,        # Z
    BLUE,       # J
    ORANGE      # L
]

# Game variables
FPS = 60
FALL_SPEED = 0.05  # Blocks per frame
SPEEDUP_FACTOR = 10  # How much faster when down is pressed
HARD_DROP_FACTOR = 100  # How much faster for hard drop

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Tetris')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

class Tetromino:
    def __init__(self, x, y, shape_idx):
        self.x = x
        self.y = y
        self.shape_idx = shape_idx
        self.shape = SHAPES[shape_idx]
        self.color = COLORS[shape_idx]
        self.rotation = 0
    
    def rotate(self, board):
        # Save current position and rotation
        old_rotation = self.rotation
        old_shape = self.shape
        
        # Rotate the shape
        self.rotation = (self.rotation + 1) % 4
        rotated_shape = []
        
        # Rotate 90 degrees clockwise
        # For 1x4 shape (I), we need special handling
        if self.shape_idx == 0:  # I shape
            if self.rotation % 2 == 0:
                rotated_shape = [[1, 1, 1, 1]]
            else:
                rotated_shape = [[1], [1], [1], [1]]
        # For 2x2 shape (O), no rotation needed
        elif self.shape_idx == 1:  # O shape
            rotated_shape = [[1, 1], [1, 1]]
        # For other shapes, rotate normally
        else:
            # Get dimensions of the shape
            height = len(old_shape)
            width = len(old_shape[0])
            
            # Create a new rotated shape
            if self.rotation % 2 == 0:  # 0 or 2 rotations (original or 180)
                for i in range(height):
                    row = []
                    for j in range(width):
                        if self.rotation == 0:
                            # Original shape
                            row.append(old_shape[i][j])
                        else:
                            # 180 degrees rotated
                            row.append(old_shape[height-1-i][width-1-j])
                    rotated_shape.append(row)
            else:  # 1 or 3 rotations (90 or 270)
                for j in range(width):
                    row = []
                    for i in range(height):
                        if self.rotation == 1:
                            # 90 degrees rotated
                            row.append(old_shape[height-1-i][j])
                        else:
                            # 270 degrees rotated
                            row.append(old_shape[i][width-1-j])
                    rotated_shape.append(row)
        
        self.shape = rotated_shape
        
        # Check if the rotation is valid
        if self.collision(board):
            # If there's a collision, revert to the old rotation
            self.rotation = old_rotation
            self.shape = old_shape
            return False
        return True
    
    def move(self, dx, dy, board):
        self.x += dx
        self.y += dy
        if self.collision(board):
            self.x -= dx
            self.y -= dy
            return False
        return True
    
    def collision(self, board):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    # Check board boundaries
                    board_x = self.x + x
                    board_y = self.y + y
                    if (board_x < 0 or board_x >= GRID_WIDTH or 
                        board_y >= GRID_HEIGHT or 
                        (board_y >= 0 and board[board_y][board_x])):
                        return True
        return False
    
    def lock(self, board):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell and 0 <= self.y + y < GRID_HEIGHT and 0 <= self.x + x < GRID_WIDTH:
                    board[self.y + y][self.x + x] = self.color
        return board
    
    def hard_drop(self, board):
        while self.move(0, 1, board):
            pass
        return True  # Indicates the piece can't move down anymore and should be locked

class Game:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece_idx = random.randint(0, len(SHAPES) - 1)
        self.game_over = False
        self.paused = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_speed = FALL_SPEED
        self.fall_progress = 0
    
    def new_piece(self):
        # Choose a random shape for the new piece
        shape_idx = getattr(self, 'next_piece_idx', random.randint(0, len(SHAPES) - 1))
        # Update next piece
        self.next_piece_idx = random.randint(0, len(SHAPES) - 1)
        # Create a new piece at the top of the board
        return Tetromino(GRID_WIDTH // 2 - 1, -2, shape_idx)
    
    def check_lines(self):
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(self.board[y]):  # If all cells in the row are filled
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0
        
        # Clear the lines
        for y in lines_to_clear:
            # Move all lines above down
            for y2 in range(y, 0, -1):
                self.board[y2] = self.board[y2 - 1][:]
            # Clear the top line
            self.board[0] = [0 for _ in range(GRID_WIDTH)]
        
        # Update score based on number of lines cleared
        lines_count = len(lines_to_clear)
        self.lines_cleared += lines_count
        
        # Adjust level based on lines cleared (every 10 lines)
        self.level = self.lines_cleared // 10 + 1
        
        # Adjust fall speed based on level
        self.fall_speed = FALL_SPEED * (1 + (self.level - 1) * 0.2)
        
        # Score calculation (more lines at once = more points per line)
        points_per_line = [0, 100, 300, 500, 800]  # 0, 1, 2, 3, 4 lines
        score_to_add = points_per_line[lines_count] * self.level
        self.score += score_to_add
        
        return lines_count
    
    def update(self, speedup=1):
        if self.game_over or self.paused:
            return
        
        # Update the falling progress
        self.fall_progress += self.fall_speed * speedup
        
        # Move the piece down if enough progress has been made
        if self.fall_progress >= 1:
            self.fall_progress = 0
            if not self.current_piece.move(0, 1, self.board):
                # If piece can't move down, lock it in place
                self.board = self.current_piece.lock(self.board)
                # Check for completed lines
                self.check_lines()
                # Create a new piece
                self.current_piece = self.new_piece()
                # Check for game over (if new piece collides immediately)
                if self.current_piece.collision(self.board):
                    self.game_over = True
    
    def draw_board(self, surface):
        # Draw background and border
        pygame.draw.rect(surface, BLACK, (BOARD_X - 2, BOARD_Y - 2, BOARD_WIDTH + 4, BOARD_HEIGHT + 4))
        pygame.draw.rect(surface, WHITE, (BOARD_X - 1, BOARD_Y - 1, BOARD_WIDTH + 2, BOARD_HEIGHT + 2), 1)
        
        # Draw the grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell_x = BOARD_X + x * GRID_SIZE
                cell_y = BOARD_Y + y * GRID_SIZE
                
                # Draw cell border
                pygame.draw.rect(surface, GRAY, (cell_x, cell_y, GRID_SIZE, GRID_SIZE), 1)
                
                # Draw filled cells
                if self.board[y][x]:
                    pygame.draw.rect(surface, self.board[y][x], (cell_x + 1, cell_y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
        
        # Draw current piece
        if self.current_piece:
            for y, row in enumerate(self.current_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        cell_x = BOARD_X + (self.current_piece.x + x) * GRID_SIZE
                        cell_y = BOARD_Y + (self.current_piece.y + y) * GRID_SIZE
                        if cell_y >= BOARD_Y:  # Only draw cells that are in the visible area
                            pygame.draw.rect(surface, self.current_piece.color, (cell_x + 1, cell_y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
    
    def draw_next_piece(self, surface):
        next_piece_x = BOARD_X + BOARD_WIDTH + 50
        next_piece_y = BOARD_Y + 100
        
        # Draw next piece text
        text = font.render("Next:", True, WHITE)
        surface.blit(text, (next_piece_x, next_piece_y - 30))
        
        # Create a temporary tetromino with the next shape
        next_shape = SHAPES[self.next_piece_idx]
        next_color = COLORS[self.next_piece_idx]
        
        # Calculate size of the next piece for centering
        shape_width = len(next_shape[0]) * GRID_SIZE
        shape_height = len(next_shape) * GRID_SIZE
        
        # Center the shape
        next_piece_x = next_piece_x + (SIDEBAR_WIDTH - shape_width) // 2 - 25
        
        # Draw the next piece
        for y, row in enumerate(next_shape):
            for x, cell in enumerate(row):
                if cell:
                    cell_x = next_piece_x + x * GRID_SIZE
                    cell_y = next_piece_y + y * GRID_SIZE
                    pygame.draw.rect(surface, next_color, (cell_x, cell_y, GRID_SIZE - 2, GRID_SIZE - 2))
    
    def draw_score(self, surface):
        sidebar_x = BOARD_X + BOARD_WIDTH + 20
        
        # Score
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (sidebar_x, BOARD_Y + 20))
        
        # Level
        level_text = font.render(f"Level: {self.level}", True, WHITE)
        surface.blit(level_text, (sidebar_x, BOARD_Y + 60))
        
        # Lines cleared
        lines_text = small_font.render(f"Lines: {self.lines_cleared}", True, WHITE)
        surface.blit(lines_text, (sidebar_x, BOARD_Y + 180))
    
    def draw_game_over(self, surface):
        if self.game_over:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            surface.blit(overlay, (0, 0))
            
            # Draw game over text
            game_over_text = font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
            surface.blit(game_over_text, text_rect)
            
            # Draw retry text
            retry_text = small_font.render("Press R to Restart", True, WHITE)
            retry_rect = retry_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            surface.blit(retry_text, retry_rect)
    
    def draw_pause(self, surface):
        if self.paused:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            surface.blit(overlay, (0, 0))
            
            # Draw pause text
            pause_text = font.render("PAUSED", True, WHITE)
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            surface.blit(pause_text, text_rect)
    
    def draw_controls(self, surface):
        controls_x = BOARD_X + BOARD_WIDTH + 20
        controls_y = BOARD_Y + 250
        
        # Title
        controls_title = small_font.render("Controls:", True, WHITE)
        surface.blit(controls_title, (controls_x, controls_y))
        
        # List of controls
        controls = [
            "Left/Right: Move",
            "Down: Soft drop",
            "Up: Rotate",
            "Space: Hard drop",
            "P: Pause",
            "Esc: Quit"
        ]
        
        for i, control in enumerate(controls):
            control_text = small_font.render(control, True, WHITE)
            surface.blit(control_text, (controls_x, controls_y + 30 + i * 20))
    
    def draw(self, surface):
        # Draw background
        surface.fill(BLACK)
        
        # Draw game elements
        self.draw_board(surface)
        self.draw_next_piece(surface)
        self.draw_score(surface)
        self.draw_controls(surface)
        
        # Draw overlays
        self.draw_game_over(surface)
        self.draw_pause(surface)

def main():
    game = Game()
    
    # Main game loop
    running = True
    speedup = 1
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            # Handle key presses
            elif event.type == KEYDOWN:
                if not game.game_over:
                    if event.key == K_LEFT:
                        game.current_piece.move(-1, 0, game.board)
                    elif event.key == K_RIGHT:
                        game.current_piece.move(1, 0, game.board)
                    elif event.key == K_DOWN:
                        speedup = SPEEDUP_FACTOR
                    elif event.key == K_UP:
                        game.current_piece.rotate(game.board)
                    elif event.key == K_SPACE:
                        # Hard drop the current piece
                        game.current_piece.hard_drop(game.board)
                        # Lock the piece in place
                        game.board = game.current_piece.lock(game.board)
                        # Check for completed lines
                        game.check_lines()
                        # Create a new piece
                        game.current_piece = game.new_piece()
                        # Check for game over (if new piece collides immediately)
                        if game.current_piece.collision(game.board):
                            game.game_over = True
                    elif event.key == K_p:
                        game.paused = not game.paused
                    elif event.key == K_ESCAPE:
                        running = False
                else:
                    if event.key == K_r:
                        game.reset()
                    elif event.key == K_ESCAPE:
                        running = False
            
            # Handle key releases
            elif event.type == KEYUP:
                if event.key == K_DOWN:
                    speedup = 1
        
        # Update game state
        game.update(speedup)
        
        # Draw the game
        game.draw(screen)
        
        # Update the display
        pygame.display.flip()
        
        # Maintain frame rate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()