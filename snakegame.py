import pygame
import numpy as np
import random
from collections import defaultdict, deque

# Game settings for rectangular grid
GRID_WIDTH = 70   # Width of the grid (number of cells horizontally)
GRID_HEIGHT = 40  # Height of the grid (number of cells vertically)
CELL_SIZE = 20    # Size of each cell in pixels
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
FOOD_REWARD = 10
STEP_PENALTY = -1
DEATH_PENALTY = -100
BOMB_PENALTY = -5  # Penalty for hitting the bomb
ACTIONS = ["up", "down", "left", "right"]

# Q-learning parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1
n_episodes = 10000  # Number of training episodes

# Initialize the Q-table
Q_table = defaultdict(lambda: {a: 0 for a in ACTIONS})

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game with Q-learning, Dynamic Gates, and Bombs (Rectangular Theme)")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
snake_color = (0, 255, 0)  # Default green snake color

# Set up font for displaying score
font = pygame.font.Font(None, 36)

# Function to generate a random color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Initialize environment
def reset():
    snake = deque([(GRID_HEIGHT // 2, GRID_WIDTH // 2)])  # Start at the center of the rectangle
    
    food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    while food in snake:
        food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    
    gate1 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    gate2 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    while gate1 == gate2 or gate1 in snake or gate2 in snake or gate1 == food or gate2 == food:
        gate1 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
        gate2 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))

    bomb = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    while bomb in snake or bomb == food or bomb == gate1 or bomb == gate2:
        bomb = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
    
    bombs = [bomb]  # Start with one bomb
    return snake, food, "up", gate1, gate2, bombs

# State representation
def get_state(snake, food):
    head_x, head_y = snake[0]
    food_x, food_y = food
    return (np.sign(food_x - head_x), np.sign(food_y - head_y))  # Direction to food

# Choose an action based on epsilon-greedy, preventing reverse direction
def choose_action(state, current_direction):
    opposite_directions = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left"
    }
    
    valid_actions = [action for action in ACTIONS if action != opposite_directions[current_direction]]
    
    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)
    else:
        return max(valid_actions, key=lambda action: Q_table[state][action])

# Q-table update
def update_q_table(state, action, reward, next_state):
    best_next_action = max(Q_table[next_state], key=Q_table[next_state].get)
    Q_table[state][action] += alpha * (reward + gamma * Q_table[next_state][best_next_action] - Q_table[state][action])

# Move snake
def move_snake(snake, direction):
    head_x, head_y = snake[0]
    if direction == "up":
        new_head = (head_x - 1, head_y)
    elif direction == "down":
        new_head = (head_x + 1, head_y)
    elif direction == "left":
        new_head = (head_x, head_y - 1)
    elif direction == "right":
        new_head = (head_x, head_y + 1)
    snake.appendleft(new_head)
    return new_head

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

# Draw the score on the screen
def draw_score(score):
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

# Draw game
def draw(snake, food, gate1, gate2, bombs, score):
    screen.fill(BLACK)
    for segment in snake:
        pygame.draw.rect(screen, snake_color, (segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (food[1] * CELL_SIZE, food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, BLUE, (gate1[1] * CELL_SIZE, gate1[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, BLUE, (gate2[1] * CELL_SIZE, gate2[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for bomb in bombs:
        pygame.draw.rect(screen, ORANGE, (bomb[1] * CELL_SIZE, bomb[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))  # Draw bombs
    
    draw_score(score)
    pygame.display.flip()

# Update gates and add a bomb at specific intervals
def update_gates_and_bombs(snake, food, gate1, gate2, bombs, last_gate_update, last_bomb_update):
    current_time = pygame.time.get_ticks()

    # Update gates every 5 seconds
    if current_time - last_gate_update > 5000:
        gate1 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
        gate2 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
        while gate1 == gate2 or gate1 in snake or gate2 in snake or gate1 == food or gate2 == food:
            gate1 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            gate2 = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
        last_gate_update = current_time

    # Update bombs every 5 seconds and add a new bomb every 8 seconds
    if current_time - last_bomb_update > 5000:
        # Add a new bomb every 8 seconds
        if current_time % 8000 < 5000:
            bomb = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            while bomb in snake or bomb == food or bomb in bombs or bomb == gate1 or bomb == gate2:
                bomb = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            bombs.append(bomb)
        last_bomb_update = current_time

    return gate1, gate2, bombs, last_gate_update, last_bomb_update

# Training phase
for episode in range(n_episodes):
    snake_color
    snake, food, direction, gate1, gate2, bombs = reset()
    state = get_state(snake, food)
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state, direction)
        direction = action
        new_head = move_snake(snake, direction)
        check_gate(snake, gate1, gate2)

        if new_head == food:
            reward = FOOD_REWARD
            food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            while food in snake or food in [gate1, gate2] or food in bombs:
                food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            snake_color = random_color()  # Change color when snake eats food
        elif new_head in bombs:
            reward = BOMB_PENALTY
            if total_reward + reward < 0:
                done = True
        else:
            snake.pop()
            reward = STEP_PENALTY

        if is_collision(snake):
            reward = DEATH_PENALTY
            done = True

        next_state = get_state(snake, food)
        update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print progress every 100 episodes
    if (episode + 1):
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

print("Training complete!")

# Testing phase to visualize the agent
for test_episode in range(5):
    snake, food, direction, gate1, gate2, bombs = reset()
    state = get_state(snake, food)
    done = False
    total_reward = 0
    score = 0
    last_gate_update = pygame.time.get_ticks()
    last_bomb_update = pygame.time.get_ticks()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        gate1, gate2, bombs, last_gate_update, last_bomb_update = update_gates_and_bombs(
            snake, food, gate1, gate2, bombs, last_gate_update, last_bomb_update
        )

        action = choose_action(state, direction)
        direction = action
        new_head = move_snake(snake, direction)
        check_gate(snake, gate1, gate2)

        if new_head == food:
            score += 1
            food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            while food in snake or food in [gate1, gate2] or food in bombs:
                food = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
            if score % 5 == 0: 
                snake_color = random_color()  # Change color when snake eats food
        elif new_head in bombs:
            total_reward += BOMB_PENALTY
            if total_reward < 0:
                print("Score went negative, game over!")
                done = True
        else:
            snake.pop()

        if is_collision(snake):
            print("Snake hit an obstacle!")
            done = True

        state = get_state(snake, food)
        score = len(snake) - 1  # Score is based on snake length minus initial segment
        draw(snake, food, gate1, gate2, bombs, score)
        pygame.time.delay(50)  # Control game speed for visualization

pygame.quit()
