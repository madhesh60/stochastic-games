import numpy as np
import random
import matplotlib.pyplot as plt

class TreasureHuntEnv:
    """
    Treasure Hunt grid environment.
    The agent searches for treasures while avoiding fire.
    """
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = 0.95
        self.start = (0, 0)
    
    def pos_to_state(self, x, y):
        return x * self.size + y
    
    def reset(self):
        """Resets the environment to the initial state."""
        self.treasures = [(random.randint(0, 4), random.randint(0, 4)) for _ in range(3)]
        self.fire = (random.randint(1, 3), random.randint(1, 3))
        self.agent_pos = self.start
        return self.pos_to_state(*self.agent_pos)
    
    def step(self, action):
        """Performs a transition step in the environment based on the selected action."""
        x, y = self.agent_pos
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        
        # Stochastic transitions
        if random.random() < 0.8:
            nx, ny = x + dx, y + dy
        else:
            nx, ny = x + dx + random.choice([-1, 0, 1]), y + dy + random.choice([-1, 0, 1])
            
        nx, ny = max(0, min(4, nx)), max(0, min(4, ny))
        self.agent_pos = (nx, ny)
        
        reward = -0.1
        if (nx, ny) in self.treasures:
            reward += 10
            self.treasures.remove((nx, ny))
        if (nx, ny) == self.fire:
            reward -= 8
            
        done = len(self.treasures) == 0
        return self.pos_to_state(nx, ny), reward, done

def train_sarsa(env, episodes=3000):
    """Trains an agent using the SARSA algorithm."""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_history = []
    alpha, epsilon = 0.1, 0.2
    
    for ep in range(episodes):
        state = env.reset()
        total_r = 0
        
        # Epsilon-greedy action selection for the initial state
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        
        while True:
            next_state, reward, done = env.step(action)
            total_r += reward
            
            # Action selection for the next state (on-policy)
            if random.random() < epsilon:
                next_action = random.randint(0, 3)
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA update rule
            Q[state, action] += alpha * (reward + env.gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            if done:
                break
        rewards_history.append(total_r)
        
        if (ep + 1) % 500 == 0:
            print(f"SARSA | Episode {ep+1} | Avg Reward: {np.mean(rewards_history[-100:]):.2f}")
            
    return Q, rewards_history

if __name__ == "__main__":
    env = TreasureHuntEnv()
    Q, rewards = train_sarsa(env)
    
    # Plot results
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    plt.title("SARSA")
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward")
    plt.show()