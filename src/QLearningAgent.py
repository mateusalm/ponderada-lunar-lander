import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((env.observation_space.shape[0], env.action_space.shape[0]))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_rate * next_max)
        self.q_table[state, action] = new_value

    def train(self, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            self.exploration_rate *= self.exploration_decay
        return total_rewards

# Função para visualização
def visualize_training(rewards):
    plt.plot(rewards)
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.title('Recompensa Total por Episódio')
    plt.show()

# Main
if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode='rgb_array') 
    agent = QLearningAgent(env)
    num_episodes = 1000
    rewards = agent.train(num_episodes)
    visualize_training(rewards)
