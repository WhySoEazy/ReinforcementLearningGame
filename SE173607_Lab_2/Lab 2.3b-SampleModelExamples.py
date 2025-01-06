# Lab 2.4 - Sample Models Example
import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.num_actions = 4  # Up, Down, Left, Right
        self.start_state = (0, 0)

    def step(self, state, action):
        # Define the dynamics of the environment
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
        return next_state

class SampleModel:
    # Initialize the environment
    def __init__(self, environment):
        self.environment = environment

    def simulate_step(self, state, action):
        # Simulate the next state based on the given action
        next_state = self.environment.step(state, action)
        return next_state

# Create a grid world environment
grid_world = GridWorld()

# Create a sample model for the grid world environment
sample_model = SampleModel(grid_world)

# Simulate a step in the environment
current_state = (0, 0)
action = np.random.choice(grid_world.num_actions)
next_state = sample_model.simulate_step(current_state, action)

# Print the simulated transition
print("Current State:", current_state)
print("Action:", action)
print("Next State:", next_state)
