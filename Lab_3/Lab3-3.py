import numpy as np
import random

class GridWorld: 
    def  __init__(self):
        self.grid_size = (3, 3)
        self.num_actions = 4 # Up, down, left, right 
        self.start_state = (0, 0)
        self.goal_state = (2, 2)
        
    def step(self, state, action):
        # Get current state
        row, col = state
        if action == 0: # Up
            row = max(0, row - 1)
        elif action == 1: # Down
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2: # Left
            col = max(0, col - 1)
        elif action == 3: # Right
            col = min(self.grid_size[1] - 1, col + 1)
        
        next_state = (row, col)
        
        reward = 0
        if next_state == self.goal_state:
            reward += 1
        
        return next_state, reward
    
class ActorCritic:
    def __init__(self, num_actions, alpha_actor, alpha_critic, gamma):
        self.num_actions = num_actions
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        self.actor_params = np.zeros((3, 3, num_actions)) # Tabular actor param
        self.critic_params = np.zeros((3, 3)) # Tabular critic param
    
    def softmax(self, x):
        """
        Compute softmax probabilities
        Prevents overflow by subtracting max value
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def select_action(self, state):
        """
        Select action probabilistically based on actor parameters
        
        Args:
            state (tuple): Current state coordinates
        
        Returns:
            int: Selected action
        """
        row, col = state
        action_preferences = self.actor_params[row, col]
        action_probs = self.softmax(action_preferences)
        
        # Sample action based on probabilities
        return np.random.choice(self.num_actions, p=action_probs)
    
    def update(self, state, action, reward, next_state):
        """
        Update actor and critic parameters
        
        Args:
            state (tuple): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (tuple): Next state
        """
        # Current state values
        row, col = state
        next_row, next_col = next_state
        
        # Compute TD error
        current_value = self.critic_params[row, col]
        next_value = self.critic_params[next_row, next_col]
        
        # TD target
        td_target = reward + self.gamma * next_value
        td_error = td_target - current_value
        
        # Update critic (value function)
        self.critic_params[row, col] += self.alpha_critic * td_error
        
        # Update actor (policy)
        # Increase probability of taken action, decrease others
        for a in range(self.num_actions):
            if a == action:
                # Increase preference for taken action
                self.actor_params[row, col, a] += self.alpha_actor * td_error
            else:
                # Decrease preferences for other actions
                self.actor_params[row, col, a] -= self.alpha_actor * td_error / (self.num_actions - 1)

def main():
    # Create GridWorld environment
    grid_world = GridWorld()

    # Create an Actor-Critic agent
    num_actions = 4 # Up, Down, Left, Right
    alpha_actor = 0.1
    alpha_critic = 0.1
    gamma = 0.9
    agent = ActorCritic(num_actions, alpha_actor, alpha_critic, gamma)

    # Train the Actor-Critic Agent
    num_episodes = 1000
    episode_rewards = []

    for episode in range(num_episodes):
        total_reward = 0
        state = grid_world.start_state
        
        while state != grid_world.goal_state:
            # Select an action
            action = agent.select_action(state)
            
            # Take the action
            next_state, reward = grid_world.step(state, action)
            
            # Update agent
            agent.update(state, action, reward, next_state)
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        # Periodic reporting
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    
    # Print final actor and critic parameters
    print("\nFinal Actor Parameters:")
    print(agent.actor_params)
    print("\nFinal Critic Parameters:")
    print(agent.critic_params)
    
    # Compute average reward
    avg_reward = np.mean(episode_rewards[-100:])
    print(f"\nAverage Reward in Last 100 Episodes: {avg_reward}")

if __name__ == "__main__":
    main()