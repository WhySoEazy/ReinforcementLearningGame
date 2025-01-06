import numpy as np
import random
class GridWorld:
    def __init__(self):
        self.grid_size = (3,4)
        self.num_actions = 4 #Up, down, left, right
        self.rewards = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        
        self.start_state = (2, 0)
    
    def step(self, state, action):
        # Define the dynamics of the environment
        
        row, col = state
        if action == 0: # Up 
            row = max(0, row - 1)
        elif action == 1: # Down
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col -1)
        elif action == 3: # Right
            col = min(self.grid_size[1] - 1, col + 1)
        
        next_state =  (row, col)
        reward = self.rewards[row, col]
        
        return next_state, reward
    
def generate_episode(grid_world):
    # Your code here
    episode = []
    state = grid_world.start_state
    
    while state != (2, 3):
        action = random.randint(0, 3)
        next_state, reward = grid_world.step(state, action)
        episode.append((next_state, reward, action))
        state = next_state
    return episode

        
    
def monteCarlo(grid_world, num_episodes, gamma = 1.0):
    # Your code here
    
    V = np.zeros(grid_world.grid_size)
    returns = {state: [] for state in [(i, j) for i in range(3) for j in range(4)]}  # To store returns for each state
    
    for _ in range(num_episodes):
        episode = generate_episode(grid_world)
        G = 0 # Initialize return 
        visited_states = set()

        # Traverse the episode backward (time step t)
        for t in reversed(range(len(episode))):
            state, reward, action = episode[t]
            G = gamma * G + reward
            
            if state not in visited_states:
                visited_states.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    
    return V
                
# Create a grid world environment
grid_world = GridWorld()

# Run Monte Carlo to estimate the state-value
num_episodes = 1000
V = monteCarlo(grid_world, num_episodes)

# Print the estimated state value function
print("Estimated State-Value Function")
print(V)    
            
        