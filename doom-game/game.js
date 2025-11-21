// Game constants
const SCREEN_WIDTH = 1024;
const SCREEN_HEIGHT = 768;
const MAP_SIZE = 16;
const TILE_SIZE = 64;
const FOV = Math.PI / 3; // 60 degrees
const MAX_DEPTH = 20;
const NUM_RAYS = SCREEN_WIDTH / 2;

// Game map (1 = wall, 0 = empty, 2 = door, 3 = exit)
const map = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,2,2,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,2,2,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
];

// Game state
let gameState = {
    running: false,
    won: false,
    gameOver: false
};

// Player
const player = {
    x: 2 * TILE_SIZE,
    y: 2 * TILE_SIZE,
    angle: 0,
    speed: 0,
    strafeSpeed: 0,
    rotationSpeed: 0,
    health: 100,
    maxHealth: 100,
    ammo: 50,
    kills: 0
};

// Enemies
const enemies = [
    { x: 8 * TILE_SIZE, y: 4 * TILE_SIZE, health: 30, maxHealth: 30, angle: 0, speed: 1, damage: 10, lastShot: 0, sprite: '👹' },
    { x: 12 * TILE_SIZE, y: 6 * TILE_SIZE, health: 30, maxHealth: 30, angle: 0, speed: 1, damage: 10, lastShot: 0, sprite: '👹' },
    { x: 4 * TILE_SIZE, y: 10 * TILE_SIZE, health: 30, maxHealth: 30, angle: 0, speed: 1, damage: 10, lastShot: 0, sprite: '👹' },
    { x: 10 * TILE_SIZE, y: 12 * TILE_SIZE, health: 30, maxHealth: 30, angle: 0, speed: 1, damage: 10, lastShot: 0, sprite: '👹' },
    { x: 13 * TILE_SIZE, y: 13 * TILE_SIZE, health: 30, maxHealth: 30, angle: 0, speed: 1, damage: 10, lastShot: 0, sprite: '👹' }
];

// Input handling
const keys = {};
let mouseX = 0;
let mouseLocked = false;

// Weapon state
const weapon = {
    bobOffset: 0,
    shootAnimOffset: 0,
    isShooting: false,
    lastShot: 0,
    shootCooldown: 300
};

// Canvas setup
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
canvas.width = SCREEN_WIDTH;
canvas.height = SCREEN_HEIGHT;

// Start screen
const startScreen = document.getElementById('startScreen');
const startButton = document.getElementById('startButton');

startButton.addEventListener('click', () => {
    startScreen.style.display = 'none';
    gameState.running = true;
    canvas.requestPointerLock();
    gameLoop();
});

// Event listeners
document.addEventListener('keydown', (e) => {
    keys[e.key.toLowerCase()] = true;

    if (e.key.toLowerCase() === 'e') {
        openDoor();
    }
});

document.addEventListener('keyup', (e) => {
    keys[e.key.toLowerCase()] = false;
});

document.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas) {
        mouseLocked = true;
        player.angle += e.movementX * 0.002;
    }
});

canvas.addEventListener('click', () => {
    if (gameState.running && !gameState.gameOver && !gameState.won) {
        canvas.requestPointerLock();
        shoot();
    }
});

document.addEventListener('pointerlockchange', () => {
    mouseLocked = document.pointerLockElement === canvas;
});

// Helper functions
function castRay(rayAngle) {
    const rayX = Math.cos(rayAngle);
    const rayY = Math.sin(rayAngle);

    let distanceToWall = 0;
    let hitWall = false;
    let wallType = 0;
    let side = 0; // 0 = vertical, 1 = horizontal

    while (!hitWall && distanceToWall < MAX_DEPTH) {
        distanceToWall += 0.1;

        const testX = player.x + rayX * distanceToWall * TILE_SIZE;
        const testY = player.y + rayY * distanceToWall * TILE_SIZE;

        const mapX = Math.floor(testX / TILE_SIZE);
        const mapY = Math.floor(testY / TILE_SIZE);

        if (mapX < 0 || mapX >= MAP_SIZE || mapY < 0 || mapY >= MAP_SIZE) {
            hitWall = true;
            distanceToWall = MAX_DEPTH;
        } else if (map[mapY][mapX] > 0) {
            hitWall = true;
            wallType = map[mapY][mapX];

            // Determine which side was hit
            const blockMidX = mapX * TILE_SIZE + TILE_SIZE / 2;
            const blockMidY = mapY * TILE_SIZE + TILE_SIZE / 2;

            const dx = testX - blockMidX;
            const dy = testY - blockMidY;

            side = Math.abs(dx) > Math.abs(dy) ? 0 : 1;
        }
    }

    return { distance: distanceToWall, wallType, side };
}

function drawWalls() {
    // Sky
    ctx.fillStyle = '#2a2a3e';
    ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT / 2);

    // Floor
    ctx.fillStyle = '#3d3d3d';
    ctx.fillRect(0, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT / 2);

    // Cast rays
    for (let x = 0; x < NUM_RAYS; x++) {
        const rayAngle = (player.angle - FOV / 2) + (x / NUM_RAYS) * FOV;
        const ray = castRay(rayAngle);

        const distance = ray.distance * Math.cos(rayAngle - player.angle); // Fix fisheye
        const wallHeight = (TILE_SIZE / distance) * 277;

        // Wall color based on type and side
        let color;
        if (ray.wallType === 1) {
            color = ray.side === 0 ? '#8B4513' : '#654321';
        } else if (ray.wallType === 2) {
            color = ray.side === 0 ? '#4a4a4a' : '#333333'; // Door
        } else if (ray.wallType === 3) {
            color = ray.side === 0 ? '#00ff00' : '#00cc00'; // Exit
        }

        // Draw wall slice
        const drawX = x * (SCREEN_WIDTH / NUM_RAYS);
        const drawY = (SCREEN_HEIGHT - wallHeight) / 2;

        ctx.fillStyle = color;
        ctx.fillRect(drawX, drawY, SCREEN_WIDTH / NUM_RAYS + 1, wallHeight);

        // Add darkness based on distance
        const darkness = Math.min(distance / MAX_DEPTH, 0.7);
        ctx.fillStyle = `rgba(0, 0, 0, ${darkness})`;
        ctx.fillRect(drawX, drawY, SCREEN_WIDTH / NUM_RAYS + 1, wallHeight);
    }
}

function drawEnemies() {
    // Sort enemies by distance (far to near)
    const sortedEnemies = enemies
        .filter(enemy => enemy.health > 0)
        .map(enemy => {
            const dx = enemy.x - player.x;
            const dy = enemy.y - player.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) - player.angle;
            return { enemy, distance, angle };
        })
        .sort((a, b) => b.distance - a.distance);

    sortedEnemies.forEach(({ enemy, distance, angle }) => {
        // Normalize angle to -PI to PI
        let normalizedAngle = angle;
        while (normalizedAngle > Math.PI) normalizedAngle -= 2 * Math.PI;
        while (normalizedAngle < -Math.PI) normalizedAngle += 2 * Math.PI;

        // Check if enemy is in FOV
        if (Math.abs(normalizedAngle) < FOV / 2 + 0.5) {
            const spriteSize = (TILE_SIZE / distance) * 277;
            const spriteX = SCREEN_WIDTH / 2 + (normalizedAngle / FOV) * SCREEN_WIDTH - spriteSize / 2;
            const spriteY = SCREEN_HEIGHT / 2 - spriteSize / 2;

            // Draw enemy sprite
            ctx.font = `${spriteSize}px Arial`;
            ctx.fillText(enemy.sprite, spriteX, spriteY + spriteSize);

            // Draw health bar
            const healthBarWidth = spriteSize;
            const healthBarHeight = 5;
            const healthPercent = enemy.health / enemy.maxHealth;

            ctx.fillStyle = '#ff0000';
            ctx.fillRect(spriteX, spriteY - 10, healthBarWidth, healthBarHeight);
            ctx.fillStyle = '#00ff00';
            ctx.fillRect(spriteX, spriteY - 10, healthBarWidth * healthPercent, healthBarHeight);
        }
    });
}

function drawWeapon() {
    const weaponY = SCREEN_HEIGHT - 200 + Math.sin(weapon.bobOffset) * 10 + weapon.shootAnimOffset;
    const weaponX = SCREEN_WIDTH / 2 - 50;

    // Draw simple gun
    ctx.fillStyle = '#444';
    ctx.fillRect(weaponX, weaponY, 100, 200);

    ctx.fillStyle = '#666';
    ctx.fillRect(weaponX + 20, weaponY, 60, 150);

    ctx.fillStyle = '#222';
    ctx.fillRect(weaponX + 35, weaponY, 30, 80);

    // Muzzle flash
    if (weapon.shootAnimOffset < 0) {
        ctx.fillStyle = '#ffff00';
        ctx.fillRect(weaponX + 25, weaponY - 20, 50, 20);
    }
}

function drawHUD() {
    // Health
    ctx.fillStyle = '#ff0000';
    ctx.font = '24px Courier New';
    ctx.fillText(`HEALTH: ${Math.max(0, player.health)}`, 20, 30);

    // Ammo
    ctx.fillStyle = '#ffff00';
    ctx.fillText(`AMMO: ${player.ammo}`, 20, 60);

    // Kills
    ctx.fillStyle = '#00ff00';
    ctx.fillText(`KILLS: ${player.kills}/${enemies.length}`, 20, 90);

    // Crosshair
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(SCREEN_WIDTH / 2 - 10, SCREEN_HEIGHT / 2);
    ctx.lineTo(SCREEN_WIDTH / 2 + 10, SCREEN_HEIGHT / 2);
    ctx.moveTo(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 10);
    ctx.lineTo(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 10);
    ctx.stroke();
}

function updatePlayer(deltaTime) {
    // Movement
    player.speed = 0;
    player.strafeSpeed = 0;

    if (keys['w']) player.speed = 3;
    if (keys['s']) player.speed = -3;
    if (keys['a']) player.strafeSpeed = -3;
    if (keys['d']) player.strafeSpeed = 3;

    // Calculate new position
    const newX = player.x + Math.cos(player.angle) * player.speed + Math.cos(player.angle + Math.PI / 2) * player.strafeSpeed;
    const newY = player.y + Math.sin(player.angle) * player.speed + Math.sin(player.angle + Math.PI / 2) * player.strafeSpeed;

    // Collision detection
    const mapX = Math.floor(newX / TILE_SIZE);
    const mapY = Math.floor(newY / TILE_SIZE);

    if (map[mapY] && map[mapY][mapX] === 0) {
        player.x = newX;
        player.y = newY;
    }

    // Check for exit
    if (map[mapY] && map[mapY][mapX] === 3 && player.kills === enemies.length) {
        gameState.won = true;
    }

    // Weapon bobbing
    if (player.speed !== 0 || player.strafeSpeed !== 0) {
        weapon.bobOffset += 0.15;
    }

    // Weapon shoot animation
    if (weapon.shootAnimOffset < 0) {
        weapon.shootAnimOffset += 5;
    }
}

function updateEnemies(deltaTime) {
    const currentTime = Date.now();

    enemies.forEach(enemy => {
        if (enemy.health <= 0) return;

        const dx = player.x - enemy.x;
        const dy = player.y - enemy.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 500) {
            // Move towards player
            const angle = Math.atan2(dy, dx);
            const newX = enemy.x + Math.cos(angle) * enemy.speed;
            const newY = enemy.y + Math.sin(angle) * enemy.speed;

            const mapX = Math.floor(newX / TILE_SIZE);
            const mapY = Math.floor(newY / TILE_SIZE);

            if (map[mapY] && map[mapY][mapX] === 0) {
                enemy.x = newX;
                enemy.y = newY;
            }

            // Shoot at player
            if (distance < 200 && currentTime - enemy.lastShot > 2000) {
                player.health -= enemy.damage;
                enemy.lastShot = currentTime;

                if (player.health <= 0) {
                    gameState.gameOver = true;
                }
            }
        }
    });
}

function shoot() {
    const currentTime = Date.now();

    if (currentTime - weapon.lastShot < weapon.shootCooldown) return;
    if (player.ammo <= 0) return;

    player.ammo--;
    weapon.lastShot = currentTime;
    weapon.shootAnimOffset = -30;

    // Check if we hit an enemy
    const centerRay = castRay(player.angle);

    enemies.forEach(enemy => {
        if (enemy.health <= 0) return;

        const dx = enemy.x - player.x;
        const dy = enemy.y - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy) / TILE_SIZE;
        const angle = Math.atan2(dy, dx);

        let angleDiff = angle - player.angle;
        while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
        while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

        // Check if enemy is in crosshair
        if (Math.abs(angleDiff) < 0.1 && distance < centerRay.distance) {
            enemy.health -= 20;

            if (enemy.health <= 0) {
                player.kills++;
            }
        }
    });
}

function openDoor() {
    const doorCheckDist = 1.5;
    const checkX = player.x + Math.cos(player.angle) * TILE_SIZE * doorCheckDist;
    const checkY = player.y + Math.sin(player.angle) * TILE_SIZE * doorCheckDist;

    const mapX = Math.floor(checkX / TILE_SIZE);
    const mapY = Math.floor(checkY / TILE_SIZE);

    if (map[mapY] && map[mapY][mapX] === 2) {
        map[mapY][mapX] = 0; // Open door
    }
}

function drawGameOver() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    ctx.fillStyle = '#ff0000';
    ctx.font = '64px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('GAME OVER', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2);

    ctx.fillStyle = '#ffffff';
    ctx.font = '24px Courier New';
    ctx.fillText('Refresh to try again', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 50);
    ctx.textAlign = 'left';
}

function drawWin() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    ctx.fillStyle = '#00ff00';
    ctx.font = '64px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('YOU WIN!', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2);

    ctx.fillStyle = '#ffffff';
    ctx.font = '24px Courier New';
    ctx.fillText('All enemies eliminated!', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 50);
    ctx.textAlign = 'left';
}

// Main game loop
let lastTime = Date.now();

function gameLoop() {
    if (!gameState.running) return;

    const currentTime = Date.now();
    const deltaTime = (currentTime - lastTime) / 1000;
    lastTime = currentTime;

    // Clear canvas
    ctx.clearRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    if (!gameState.gameOver && !gameState.won) {
        // Update
        updatePlayer(deltaTime);
        updateEnemies(deltaTime);

        // Draw
        drawWalls();
        drawEnemies();
        drawWeapon();
        drawHUD();
    } else if (gameState.gameOver) {
        drawGameOver();
    } else if (gameState.won) {
        drawWin();
    }

    requestAnimationFrame(gameLoop);
}
