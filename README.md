<<<<<<< HEAD
# claude_minis
Experiments with Claude Code: few-shot projects or "minis"
=======
# Space Shooter

A simple space shooter game built with Pygame.

## Description

In this game, you control a spaceship at the bottom of the screen and must shoot down enemy ships that appear from the top of the screen. The game ends if an enemy ship collides with your ship.

## Controls

- **Left Arrow**: Move ship left
- **Right Arrow**: Move ship right
- **Space**: Shoot
- **R**: Restart game (after game over)
- **Q**: Quit game (after game over)

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## Installation & Setup

### 1. Clone the repository

```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Create and activate a virtual environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

With the virtual environment activated, run:

```bash
python3 space_shooter.py
# OR on Windows
python space_shooter.py
```

## Game Features

- Simple triangular player ship that moves horizontally
- Enemy ships spawn randomly from the top of the screen
- Shooting mechanics with collision detection
- Score tracking
- Game over screen with restart option
- Starry background

## For Developers

If you want to modify the game, the main file `space_shooter.py` contains all the game logic including:
- Player and enemy movement
- Collision detection
- Drawing functions
- Game state management

To update the requirements file after adding new dependencies:
```bash
pip freeze > requirements.txt
```
>>>>>>> b41b65e (CCE-01 space_shooter game)
