# Lab 2.3 - Example Distribution model
import numpy as np
import random

class Bandit:
    def __init__(self, true_means):
        self.true_means = true_means

    def pull_arm(self, arm):
        return np.random.normal(self.true_means[arm], 1)

class DistributionModel:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.mean_rewards = np.zeros(num_arms)
        self.variance_rewards = np.ones(num_arms)
        self.counts = np.zeros(num_arms)  # To track the number of times each arm is pulled

    # Update mean and variance of rewards for the selected arm
    def update_distribution(self, arm, reward):
        # Increment the count for the selected arm
        self.counts[arm] += 1
        n = self.counts[arm]
        
        # Update the mean using incremental formula
        old_mean = self.mean_rewards[arm]
        new_mean = old_mean + (reward - old_mean) / n
        self.mean_rewards[arm] = new_mean
        
        # Update the variance using Welford's method
        if n > 1:
            self.variance_rewards[arm] = ((n - 2) / (n - 1)) * self.variance_rewards[arm] + ((reward - old_mean) ** 2) / n

# Define the true means of the bandit arms
true_means = [1.0, 2.0]

# Create a bandit environment with the true means
bandit = Bandit(true_means)

# Create a distribution model for the bandit
distribution_model = DistributionModel(num_arms=len(true_means))

# Pull arms and update distribution model
num_pulls = 1000
for _ in range(num_pulls):
    # Randomly select an arm to pull
    arm = random.randint(0, len(true_means) - 1)
    
    # Pull the selected arm and observe reward
    reward = bandit.pull_arm(arm)
    
    # Update distribution model
    distribution_model.update_distribution(arm, reward)

# Print the updated distribution model
print("Updated Distribution Model:")
print("Mean Rewards:", distribution_model.mean_rewards)
print("Variance of Rewards:", distribution_model.variance_rewards)
