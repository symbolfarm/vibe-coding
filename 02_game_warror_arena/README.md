# Warrior Arena

A simple NES Zelda-inspired arena combat game built with Pygame.

## Game Description

Control a warrior in a square arena and fight enemies that wander around. Swing your sword to defeat enemies and earn points. The game continues until you run out of health.

## Controls

- **Movement**: Arrow keys or WASD
- **Sword Attack**: Space or J
- **Restart Game**: R (after game over)
- **Quit Game**: Q (after game over)

## Features

- Checkerboard arena layout
- Player with directional movement and sword attacks
- Enemies that wander randomly and follow the player
- Health and score tracking
- Game over screen with restart option

## Installation

1. Ensure you have Python 3.x installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Game

```
python warrior_arena.py
```

## Game Mechanics

- Player starts with 5 health points
- Enemies spawn from the edges of the screen
- Defeating an enemy with your sword earns 10 points
- Colliding with an enemy costs 1 health point
- Game ends when health reaches zero
