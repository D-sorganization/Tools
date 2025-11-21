# Doom-like FPS Game

A classic first-person shooter game built with HTML5 Canvas and JavaScript, featuring raycasting rendering (the same technique used in the original Doom).

## Features

- **3D Raycasting Engine**: Real-time 3D rendering using raycasting algorithm
- **First-Person Controls**:
  - WASD for movement
  - Mouse for looking around
  - Click to shoot
  - E to open doors
- **Enemy AI**: Enemies track and attack the player
- **Combat System**: Health, ammo, and damage mechanics
- **Interactive Environment**: Doors that can be opened
- **Win/Lose Conditions**: Eliminate all enemies to reach the exit and win

## How to Play

1. Open `index.html` in a modern web browser
2. Click "START GAME"
3. Use WASD to move, mouse to look around
4. Click to shoot enemies (you have 50 bullets)
5. Press E to open doors (gray walls)
6. Kill all 5 enemies to unlock the exit (green wall)
7. Don't let your health reach 0!

## Game Elements

- **Brown Walls**: Solid obstacles
- **Gray Walls**: Doors (press E to open)
- **Green Wall**: Exit (only accessible after killing all enemies)
- **Red Monsters**: Enemies that chase and shoot at you
- **Health**: Starts at 100, decreases when enemies attack
- **Ammo**: Starts at 50 bullets

## Technical Details

- Built with vanilla JavaScript and HTML5 Canvas
- Implements raycasting for 3D perspective
- Real-time enemy AI with pathfinding
- Collision detection
- Weapon animations and recoil
- HUD display

## Controls Summary

| Key | Action |
|-----|--------|
| W | Move forward |
| S | Move backward |
| A | Strafe left |
| D | Strafe right |
| Mouse | Look around |
| Click | Shoot |
| E | Open door |

## Tips

- Conserve ammo - you only have 50 bullets for 5 enemies
- Each enemy takes multiple shots to kill
- Keep moving to avoid enemy fire
- Enemies deal 10 damage per hit
- Use doors strategically to control enemy approach

Enjoy the game!
