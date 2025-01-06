import pygame
import numpy as np
import random
from collections import defaultdict, deque

# Game settings
GRID_SIZE = 10  # 10x10 grid
CELL_SIZE = 30  # Size of each cell in pixels
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FOOD_REWARD = 10
STEP_PENALTY = -1
DEATH_PENALTY = -100
ACTIONS = ["up", "down", "left", "right"]

# Q-learning parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1
n_episodes = 2000  # Number of training episodes

# Initialize the Q-table
Q_table = defaultdict(lambda: {a: 0 for a in ACTIONS})

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Snake Game with Q-learning")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Initialize environment
def reset():
    snake = deque([(GRID_SIZE // 2, GRID_SIZE // 2)])  # Start at the center
    food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    while food in snake:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    return snake, food, "up"

# State representation
def get_state(snake, food):
    head_x, head_y = snake[0]
    food_x, food_y = food
    return (np.sign(food_x - head_x), np.sign(food_y - head_y))  # Direction to food

# Choose an action based on epsilon-greedy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    return max(Q_table[state], key=Q_table[state].get)

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
        head[0] < 0 or head[0] >= GRID_SIZE or
        head[1] < 0 or head[1] >= GRID_SIZE or
        head in list(snake)[1:]
    )

# Draw game
def draw(snake, food):
    screen.fill(BLACK)
    for segment in snake:
        pygame.draw.rect(screen, GREEN, (segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (food[1] * CELL_SIZE, food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()

# Training loop
for episode in range(n_episodes):
    snake, food, direction = reset()
    state = get_state(snake, food)
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)
        direction = action
        new_head = move_snake(snake, direction)

        # Reward for eating food
        if new_head == food:
            reward = FOOD_REWARD
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            while food in snake:
                food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        else:
            snake.pop()  # Remove tail if not eating food
            reward = STEP_PENALTY

        # Check for collision
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
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

print("Training complete!")

# Visualize the trained agent
for test_episode in range(5):  # Run 5 test episodes
    snake, food, direction = reset()
    state = get_state(snake, food)
    done = False
    total_reward = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Choose best action from Q-table
        action = max(Q_table[state], key=Q_table[state].get)
        direction = action
        new_head = move_snake(snake, direction)

        if new_head == food:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            while food in snake:
                food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        else:
            snake.pop()  # Remove tail if not eating food

        # Check for collision
        if is_collision(snake):
            print("Snake hit an obstacle!")
            done = True

        state = get_state(snake, food)
        draw(snake, food)
        pygame.time.delay(100)  # Control game speed for visualization

pygame.quit()