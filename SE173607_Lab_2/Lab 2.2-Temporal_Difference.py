import numpy as np
import random

class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.num_actions = 4  # Up, Down, Left, Right
        self.goal_state = (2, 2)
        self.start_state = (2, 0)
        self.rewards = np.zeros(self.grid_size)
        self.rewards[self.goal_state] = 1  # Assign reward of 1 to the goal state
        
    # Return the next state and reward
    def step(self, state, action):
        row, col = state
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.grid_size[1] - 1, col + 1)
        
        next_state = (row, col)
        reward = self.rewards[row, col]
        
        return next_state, reward

def td_learning(grid_world, num_episodes, alpha, gamma):
    # Initialize value function V(s) with zeros
    V = np.zeros(grid_world.grid_size)
    
    for _ in range(num_episodes):
        # Start at the initial state
        state = grid_world.start_state
        
        # Loop until reaching the goal state
        while state != grid_world.goal_state:
            # Choose an action using epsilon-greedy policy
            epsilon = 0.1
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, grid_world.num_actions - 1)  # Exploration
            else:
                action_values = [V[grid_world.step(state, a)[0]] for a in range(grid_world.num_actions)]
                action = np.argmax(action_values)  # Exploitation
            
            # Take action and get next state and reward
            next_state, reward = grid_world.step(state, action)
            
            # Update value function using the TD(0) formula
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            
            # Move to the next state
            state = next_state

    return V

# Create a grid world environment
grid_world = GridWorld()

# TD Learning Parameters
num_episodes = 1000
alpha = 0.1
gamma = 0.9

# Run TD Learning to estimate the value function
values = td_learning(grid_world, num_episodes, alpha, gamma)

# Print the resulting value function
print("Value Function")
print(values)
