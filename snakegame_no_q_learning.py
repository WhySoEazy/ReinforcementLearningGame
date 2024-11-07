import pygame
import random
from collections import deque
import time
import math

# Game settings for rectangular frame
GRID_WIDTH = 70   # 60 cells wide
GRID_HEIGHT = 40  # 30 cells high
CELL_SIZE = 20    # Size of each cell in pixels
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
ACTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
DIRECTIONS = ["up", "down", "left", "right"]

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game with Moving Gates, Bombs, and Second Snake")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)  # Color for bombs
PURPLE = (255, 0, 255)  # Color for second snake

# Font for score and messages
font = pygame.font.Font(None, 36)  # Default font, size 36

# Function to generate a random color
def get_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

# Initialize environment
def reset():
    # Set up the player's snake
    player_snake = deque([(GRID_HEIGHT // 2, GRID_WIDTH // 2)])  # Start at the center
    player_color = GREEN  # Initial color for player's snake
    
    # Set up food in a random position, avoiding the player's snake's starting position
    food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    while food in player_snake:
        food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    
    # Set up initial bomb list with one bomb
    bombs = [generate_bomb(player_snake, food)]
    
    # Set up initial random positions for gate1 and gate2
    gate1, gate2 = generate_gates(player_snake, food, bombs)
    
    # Set up the second snake with the same length and random position
    second_snake = deque([(random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))])
    while second_snake[0] in player_snake or second_snake[0] == food:
        second_snake[0] = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    second_snake_direction = random.choice(DIRECTIONS)
    
    return player_snake, player_color, food, bombs, "up", gate1, gate2, second_snake, second_snake_direction

def generate_gates(snake, food, bombs):
    # Generate positions for two gates, ensuring they don't overlap with the snake, food, or bombs
    gate1 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    gate2 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    while gate1 == gate2 or gate1 in snake or gate2 in snake or gate1 == food or gate2 == food or gate1 in bombs or gate2 in bombs:
        gate1 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
        gate2 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    return gate1, gate2

def generate_bomb(snake, food):
    # Generate a bomb in a random position, ensuring it doesn't overlap with the snake or food
    bomb = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    while bomb in snake or bomb == food:
        bomb = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    return bomb

# Move snake
def move_snake(snake, direction):
    head_x, head_y = snake[0]
    dx, dy = ACTIONS[direction]
    new_head = (head_x + dy, head_y + dx)
    snake.appendleft(new_head)
    return new_head

# Move second snake randomly within bounds
def move_second_snake(snake, direction):
    head_x, head_y = snake[0]
    dx, dy = ACTIONS[direction]
    new_head = (head_x + dy, head_y + dx)
    
    # Check if the new head is out of bounds
    if new_head[0] < 0 or new_head[0] >= GRID_HEIGHT or new_head[1] < 0 or new_head[1] >= GRID_WIDTH:
        # If out of bounds, pick a new random direction
        direction = random.choice(DIRECTIONS)
        dx, dy = ACTIONS[direction]
        new_head = (head_x + dy, head_y + dx)
    else:
        # Occasionally change direction randomly to avoid edge-hugging
        if random.random() < 0.2:  # 20% chance to change direction
            direction = random.choice(DIRECTIONS)

    snake.appendleft(new_head)
    return new_head, direction

# Check for collisions
def is_collision(snake):
    head = snake[0]
    return (
        head[0] < 0 or head[0] >= GRID_HEIGHT or
        head[1] < 0 or head[1] >= GRID_WIDTH or
        head in list(snake)[1:]
    )

# Check for gate teleportation
def check_gate(snake, gate1, gate2):
    head = snake[0]
    if head == gate1:
        snake[0] = gate2  # Teleport to gate2 if entering gate1
    elif head == gate2:
        snake[0] = gate1  # Teleport to gate1 if entering gate2

# Calculate distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Draw game
def draw(snake, player_color, food, bombs, gate1, gate2, score, second_snake, show_hello):
    screen.fill(BLACK)
    for segment in snake:
        pygame.draw.rect(screen, player_color, (segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (food[1] * CELL_SIZE, food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for bomb in bombs:
        pygame.draw.rect(screen, YELLOW, (bomb[1] * CELL_SIZE, bomb[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))  # Draw each bomb
    pygame.draw.rect(screen, BLUE, (gate1[1] * CELL_SIZE, gate1[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, BLUE, (gate2[1] * CELL_SIZE, gate2[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw the second snake
    for segment in second_snake:
        pygame.draw.rect(screen, PURPLE, (segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Render and draw score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))  # Draw score in the top-left corner

    # Display "Hello" if the two snakes are close to each other
    if show_hello:
        hello_text = font.render("Hello", True, WHITE)
        screen.blit(hello_text, (SCREEN_WIDTH // 2 - hello_text.get_width() // 2, SCREEN_HEIGHT // 2 - hello_text.get_height() // 2))

    pygame.display.flip()

# Function to display "Game Over" message
def display_game_over():
    game_over_text = font.render("Game Over", True, RED)
    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(3000)  # Wait for 3 seconds

# Main game loop
player_snake, player_color, food, bombs, direction, gate1, gate2, second_snake, second_snake_direction = reset()
score = 0  # Initialize score
done = False
clock = pygame.time.Clock()
last_gate_change = time.time()  # Timer for gate position change
last_bomb_change = time.time()  # Timer for bomb position change
last_bomb_addition = time.time()  # Timer for adding new bombs

while not done:
    current_time = time.time()
    
    # Check if 10 seconds have passed to change gate positions
    if current_time - last_gate_change >= 10:
        gate1, gate2 = generate_gates(player_snake, food, bombs)
        last_gate_change = current_time  # Reset the timer for gate change
    
    # Check if 5 seconds have passed to change bomb positions
    if current_time - last_bomb_change >= 5:
        bombs = [generate_bomb(player_snake, food) for _ in bombs]  # Re-generate positions for each bomb
        last_bomb_change = current_time  # Reset the timer for bomb change
    
    # Check if 8 seconds have passed to add a new bomb
    if current_time - last_bomb_addition >= 8:
        bombs.append(generate_bomb(player_snake, food))  # Add a new bomb
        last_bomb_addition = current_time  # Reset the timer for adding new bombs
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != "down":
                direction = "up"
            elif event.key == pygame.K_DOWN and direction != "up":
                direction = "down"
            elif event.key == pygame.K_LEFT and direction != "right":
                direction = "left"
            elif event.key == pygame.K_RIGHT and direction != "left":
                direction = "right"
    
    new_head = move_snake(player_snake, direction)
    check_gate(player_snake, gate1, gate2)  # Check if the snake should teleport between gates

    # Move the second snake randomly within bounds
    second_head, second_snake_direction = move_second_snake(second_snake, second_snake_direction)
    if len(second_snake) > len(player_snake):
        second_snake.pop()  # Ensure it matches the length of the player snake

    # Check proximity greeting
    show_hello = False
    if distance(player_snake[0], second_snake[0]) < 3 / CELL_SIZE:  # Convert pixels to grid units
        show_hello = True

    # Check if player snake eats the food
    if new_head == food:
        score += 1  # Increase score by 1 for eating food
        
        # Change color every 10 points
        if score % 10 == 0:
            player_color = get_random_color()
        
        food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
        while food in player_snake or food in [gate1, gate2] or food in bombs:  # Ensure food is not placed on gates or bombs
            food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    else:
        player_snake.pop()  # Remove tail if not eating food

    # Check for collision with any bomb
    if new_head in bombs:
        score -= 5  # Deduct 5 points for hitting a bomb
        print("Hit a bomb! Score decreased by 5.")
        if score < 0:  # Check if score is negative
            print("Score is negative. Game over!")
            display_game_over()
            done = True

    # Check for collision with wall or itself
    if is_collision(player_snake):
        print("Snake hit an obstacle! Game over!")
        display_game_over()
        done = True

    # Check for collision with second snake
    if new_head in second_snake:
        print("Snake collided with the second snake! Game over!")
        display_game_over()
        done = True

    draw(player_snake, player_color, food, bombs, gate1, gate2, score, second_snake, show_hello)
    clock.tick(10)  # Control game speed

pygame.quit()