import numpy as np
import random
import matplotlib.pyplot as plt

# ================== ENVIRONMENT ==================
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
            nx = x + dx + random.choice([-1, 0, 1])
            ny = y + dy + random.choice([-1, 0, 1])
            
        nx = max(0, min(4, nx))
        ny = max(0, min(4, ny))
        self.agent_pos = (nx, ny)
        
        reward = -0.1
        if (nx, ny) in self.treasures:
            reward += 10
            self.treasures.remove((nx, ny))
        if (nx, ny) == self.fire:
            reward -= 8
            
        done = len(self.treasures) == 0
        return self.pos_to_state(nx, ny), reward, done

# ================== TEST FUNCTION ==================
def evaluate_agent(Q, env, episodes=200, epsilon=0.0):
    """Evaluates the trained agent's policy in the environment."""
    total_rewards = []
    success_count = 0
    steps_list = []
    
    for _ in range(episodes):
        state = env.reset()
        total_r = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            # Action selection using greedy policy (no exploration during evaluation)
            action = np.argmax(Q[state])
            next_state, reward, done = env.step(action)
            total_r += reward
            steps += 1
            state = next_state
            
        total_rewards.append(total_r)
        steps_list.append(steps)
        if done and len(env.treasures) == 0:
            success_count += 1
            
    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "success_rate": success_count / episodes * 100,
        "avg_steps": np.mean(steps_list)
    }

# ================== TRAINING FUNCTIONS ==================

def train_q_learning(env, episodes=2500):
    """Trains an agent using the Q-Learning algorithm."""
    Q = np.zeros((env.n_states, env.n_actions))
    alpha, epsilon = 0.1, 0.2
    for ep in range(episodes):
        state = env.reset()
        while True:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[state])
                
            next_state, reward, done = env.step(action)
            Q[state, action] += alpha * (reward + env.gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if done:
                break
    return Q

def train_double_q(env, episodes=2500):
    """Trains an agent using the Double Q-Learning algorithm."""
    Q1 = np.zeros((env.n_states, env.n_actions))
    Q2 = np.zeros((env.n_states, env.n_actions))
    alpha, epsilon = 0.1, 0.15
    for ep in range(episodes):
        state = env.reset()
        while True:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q1[state] + Q2[state])
                
            next_state, reward, done = env.step(action)
            
            if random.random() < 0.5:
                a_star = np.argmax(Q1[next_state])
                Q1[state, action] += alpha * (reward + env.gamma * Q2[next_state, a_star] - Q1[state, action])
            else:
                a_star = np.argmax(Q2[next_state])
                Q2[state, action] += alpha * (reward + env.gamma * Q1[next_state, a_star] - Q2[state, action])
            state = next_state
            if done:
                break
    return Q1 + Q2

def train_sarsa(env, episodes=2500):
    """Trains an agent using the SARSA algorithm."""
    Q = np.zeros((env.n_states, env.n_actions))
    alpha, epsilon = 0.1, 0.2
    for ep in range(episodes):
        state = env.reset()
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
            
        while True:
            next_state, reward, done = env.step(action)
            if random.random() < epsilon:
                next_action = random.randint(0, 3)
            else:
                next_action = np.argmax(Q[next_state])
                
            Q[state, action] += alpha * (reward + env.gamma * Q[next_state, next_action] - Q[state, action])
            state = next_state
            action = next_action
            if done:
                break
    return Q

# ================== MAIN COMPARISON ==================
if __name__ == "__main__":
    env = TreasureHuntEnv()
    n_train = 3000
    n_test = 300
    
    print("Training all three algorithms...\n")
    
    print("1. Training Standard Q-Learning...")
    Q_ql = train_q_learning(env, n_train)
    
    print("2. Training Double Q-Learning...")
    Q_dq = train_double_q(env, n_train)
    
    print("3. Training SARSA...")
    Q_sarsa = train_sarsa(env, n_train)
    
    print("\nEvaluating all agents...\n")
    
    results_ql = evaluate_agent(Q_ql, env, n_test)
    results_dq = evaluate_agent(Q_dq, env, n_test)
    results_sarsa = evaluate_agent(Q_sarsa, env, n_test)
    
    print("="*60)
    print("FINAL ACCURACY / PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Algorithm':<20} {'Avg Reward':<12} {'Success Rate':<15} {'Avg Steps':<12} {'Std Reward':<12}")
    print("-"*60)
    print(f"{'Q-Learning':<20} {results_ql['avg_reward']:<12.2f} {results_ql['success_rate']:<15.1f}% {results_ql['avg_steps']:<12.1f} {results_ql['std_reward']:<12.2f}")
    print(f"{'Double Q-Learning':<20} {results_dq['avg_reward']:<12.2f} {results_dq['success_rate']:<15.1f}% {results_dq['avg_steps']:<12.1f} {results_dq['std_reward']:<12.2f}")
    print(f"{'SARSA':<20} {results_sarsa['avg_reward']:<12.2f} {results_sarsa['success_rate']:<15.1f}% {results_sarsa['avg_steps']:<12.1f} {results_sarsa['std_reward']:<12.2f}")
    print("="*60)

    # Plot Comparison
    algorithms = ['Q-Learning', 'Double Q', 'SARSA']
    rewards = [results_ql['avg_reward'], results_dq['avg_reward'], results_sarsa['avg_reward']]
    success = [results_ql['success_rate'], results_dq['success_rate'], results_sarsa['success_rate']]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(algorithms, rewards)
    plt.title("Average Reward Comparison")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    plt.bar(algorithms, success)
    plt.title("Success Rate (%)")
    plt.ylabel("Success Rate")
    plt.show()