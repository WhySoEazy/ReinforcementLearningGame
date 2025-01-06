import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier

class GridWorld:
    def __init__(self):
        self.grid_size = (2,2)
        self.num_actions = 4 # Up, down, left, right
        self.start_state = (0, 0)
        self.goal_state = (2, 2)
        
    def step(self, state, action):
        
        # Define the dynamics of the environment
        row, col =  state
        
        if action == 0: # Up 
            row =  max(0, row - 1)
        elif action == 1: # Down 
            row =  min(self.grid_size[0] - 1, row + 1)
        elif action == 2: # Left 
            col = max(0, col - 1)
        elif action == 3: # Right
            col = min(self.grid_size[1] - 1, col + 1)
        
        next_state = (row, col)
        
        reward = 0 
        
        if next_state == self.goal_state:
            reward  = 1 # Reward + 1 upon reaching the goal state
        
        return next_state, reward

def generate_training_data(grid_world, num_samples):
    X = np.zeros((num_samples, 2)) # state features
    Y = np.zeros((num_samples,)) # Actions
        
    for i in range(num_samples):
        state = (np.random.randint(grid_world.grid_size[0]),
                np.random.randint(grid_world.grid_size[1]))
        action = np.random.randint(grid_world.num_actions)
        next_state, _ = grid_world.step(state, action)
        X[i] = state
        Y[i] = action
    
    return X, Y
    
grid_world = GridWorld()

# Generate training data
num_samples = 10000
X_train, y_train = generate_training_data(grid_world, num_samples)

# Model 

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Evaluate the learned policy, return total_reward

def evaluate_policy(grid_world, model):
    total_reward = 0
    current_state = (0, 0)  # Example start state
    for _ in range(100):  # Limit steps for evaluation
        # Predict action based on current state
        action = model.predict([list(current_state)])[0]
        
        # Move to next state based on action (simplified for example purposes)
        # Update current_state based on the action
        if action == 'up' and current_state[0] > 0:
            current_state = (current_state[0] - 1, current_state[1])
        elif action == 'down' and current_state[0] < grid_world.size[0] - 1:
            current_state = (current_state[0] + 1, current_state[1])
        elif action == 'left' and current_state[1] > 0:
            current_state = (current_state[0], current_state[1] - 1)
        elif action == 'right' and current_state[1] < grid_world.size[1] - 1:
            current_state = (current_state[0], current_state[1] + 1)

        # Calculate reward (1 if at goal state, 0 otherwise)
        reward = 1 if current_state == grid_world.goal_state else 0
        total_reward += reward
    
    return total_reward

total_reward = evaluate_policy(grid_world, model)
print(total_reward)



    

        
    