import numpy as np 
import random


class GridWorld:
    def __init__(self):
        self.grid_size = (4, 4)
        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.num_actions = 4 #Up, Down, Left, Right
        
    
    def step(self, state, action):
        # Define the dynamics of the environment
        row, col = state
        
        if action == 0: # Up
            row = max(0, row - 1)
        elif action == 1: # Down
            row = min(self.grid_size[0] -1, row + 1)
        elif action == 2: # Left
            col = max(0, col - 1)
        elif action == 3: # Right
            col = min(self.grid_size[1] - 1, col + 1)
            
        next_state = (row, col)
        reward = 0
        if next_state == self.goal_state:
            reward = 1
        return next_state, reward

# Epsilon-greddy policy for action selection

def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q[state])) # Explore: choose a random action
    else:
        return np.argmax(Q[state]) # Exploit: Choose the action with the highest Q
    
# SARSA algorithm to learn the optimal policy 

def sarsa(grid_world, num_episodes, alpha, gamma, epsilon):
    
    Q = np.zeros((grid_world.grid_size[0], grid_world.grid_size[1],  grid_world.num_actions))
    
    for episodes in range(num_episodes):
        state = grid_world.start_state
        action = epsilon_greedy_policy(Q, state, epsilon)
        
        while state != grid_world.goal_state:
            next_state, reward = grid_world.step(state, action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            # UPdate Q-value using SARSA update rule
            
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
    return Q
            
grid_world = GridWorld()
num_episodes =  1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Learn the optimal policy 

Q = sarsa(grid_world, num_episodes,  alpha, gamma, epsilon)

print("Learned Q-value Function:")
print(Q)
