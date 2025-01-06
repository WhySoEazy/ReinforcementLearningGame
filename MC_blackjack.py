import gym
import numpy as np
from collections import defaultdict

# Initialize environment
env = gym.make('Blackjack-v1')

# Initialize Q-values, returns, and policy
Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Q-value for each state-action pair
returns_sum = defaultdict(float)  # Total return for each state-action pair
returns_count = defaultdict(float)  # Count of how many times each state-action pair is visited

# Hyperparameters
epsilon = 0.1  # Exploration rate
gamma = 1.0    # Discount factor

# Define epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

# Generate episode
def generate_episode(policy, epsilon):
    episode = []
    state = env.reset()  # Older gym versions return only state, no info
    done = False
    
    while not done:
        action = policy(state, epsilon)  # Choose action using ε-greedy policy
        next_state, reward, done, info = env.step(action)  # Older gym returns 4 values from step()
        episode.append((state, action, reward))
        state = next_state
        print(episode)
    
    return episode

# Update Q-values based on the episode
def update_Q(episode, gamma):
    G = 0  # Initialize the return
    visited_state_action_pairs = set()  # Keep track of visited state-action pairs to avoid duplicates
    
    # Work backwards through the episode
    for state, action, reward in reversed(episode):
        G = gamma * G + reward  # Calculate the cumulative return
        
        if (state, action) not in visited_state_action_pairs:
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]  # Update Q-value
            visited_state_action_pairs.add((state, action))

# Train the agent using Monte Carlo control
def train_blackjack_MC(episodes, epsilon=0.1, gamma=1.0):
    for i in range(episodes):
        # Generate an episode following the ε-greedy policy
        episode = generate_episode(epsilon_greedy_policy, epsilon)
        
        # Update Q-values using the generated episode
        update_Q(episode, gamma)
        
    print("Training completed!")

# Test the agent's performance
def test_blackjack_MC(episodes):
    wins = 0
    losses = 0
    draws = 0

    for _ in range(episodes):
        state = env.reset()  # Older gym returns only state
        done = False

        while not done:
            action = np.argmax(Q[state])  # Always exploit the learned Q-values
            state, reward, done, info = env.step(action)  # Older gym returns 4 values

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"Results after {episodes} episodes: {wins} wins, {losses} losses, {draws} draws")
    print('Win rate:' , (wins / episodes) * 100 , '%')

# Train the agent
train_blackjack_MC(5000)  # Train for 500,000 episodes

# Test the agent
test_blackjack_MC(100)  # Test for 1,000 episodes