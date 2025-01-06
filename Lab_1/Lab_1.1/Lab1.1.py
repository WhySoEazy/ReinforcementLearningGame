import numpy as np
import random 

# Define a class

class EpsilonGreedyAgent:
    def __init__(self, num_actions, epsilon = 0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.action_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        
    # Randomly choose an action for exploration
    # Choose greddy action for exploration
    
    def select_action(self):
        # Your code here
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.action_values)
        return action
    
    # Update action-value estimate using incremental update rule
    def update_value(self, action, reward):
        # Your code here
        self.action_counts[action] += 1
        
        # Incremental update formula for Q-values (action avlue estimated)
        alpha = 1.0/ self.action_counts[action]
        self.action_values[action] += alpha * (reward - self.action_values[action])
        
# Create a simple multi-armed bandit environment
class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_action_values = np.random.normal(0, 1, num_arms)
    
    # Reward is sampled from a normal distribution with mean true action value and unit variance
    def get_reward(self, action):
        # Your code here
        return np.random.normal(self.true_action_values[action], 1.0)
# Initialize the environment and agent 
num_arms = 10
num_steps = 1000
agent = EpsilonGreedyAgent(num_arms)

# Interation loop
bandit = MultiArmedBandit(num_arms)
total_rewards = 0
for step in range(num_steps):
    action = agent.select_action() # Agent selects an action 
    reward = bandit.get_reward(action) # Environment gives rewards based on action
    agent.update_value(action, reward) # Agent updates its action value estimate
    total_rewards += reward
print(f"Total rewards obtained: {total_rewards}")
print(f"Estimated action values: {agent.action_values}")

