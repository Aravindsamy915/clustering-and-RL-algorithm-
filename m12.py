import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
# Environment parameters
num_lanes = 4
max_queue_length = 10
num_actions = 4  # Four possible actions (green light for each of the 4 lanes)
# SARSA parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration-exploitation balance
# Initialize Q-table
Q = np.zeros((max_queue_length + 1, max_queue_length + 1, max_queue_length + 1, max_queue_length + 1, num_actions))
# Function to choose an action using epsilon-greedy strategy
def choose_action(state, epsilon):
if np.random.uniform(0, 1) < epsilon:
return random.randint(0, num_actions - 1)  # Explore
else:
return np.argmax(Q[state])  # Exploit
# Simulate environment step
def step(state, action):
next_state = list(state)
reward = 0
# Simulate the traffic control effect
for i in range(num_lanes):
if i == action:  # Green light reduces queue length
next_state[i] = max(0, state[i] - random.randint(1, 3))
else:  # Red light increases queue length
next_state[i] = min(max_queue_length, state[i] + random.randint(0, 2))

reward = -sum(next_state)  # Reward is negative of total queue length to minimize it
return tuple(next_state), reward
# Function to create a synthetic dataset
def create_synthetic_traffic_dataset(num_samples):
data = []
for _ in range(num_samples):
# Random initial state
state = (random.randint(0, max_queue_length), random.randint(0, max_queue_length),
random.randint(0, max_queue_length), random.randint(0, max_queue_length))
action = random.randint(0, num_actions - 1)
# Get next state and reward
next_state, reward = step(state, action)
# Add to dataset
data.append({
'queue_lane_1': state[0],
'queue_lane_2': state[1],
'queue_lane_3': state[2],
'queue_lane_4': state[3],
'action': action,
'next_queue_lane_1': next_state[0],
'next_queue_lane_2': next_state[1],
'next_queue_lane_3': next_state[2],
'next_queue_lane_4': next_state[3],
'reward': reward
})
return pd.DataFrame(data)
# SARSA algorithm for traffic control using dataset
def sarsa_traffic_using_dataset(dataset):
rewards = []

# Loop through the dataset
for idx, row in dataset.iterrows():
state = (row['queue_lane_1'], row['queue_lane_2'], row['queue_lane_3'], row['queue_lane_4'])
action = row['action']
next_state = (row['next_queue_lane_1'], row['next_queue_lane_2'], row['next_queue_lane_3'], row['next_queue_lane_4'])
reward = row['reward']
next_action = choose_action(next_state, epsilon)
# Update Q-value using SARSA formula
Q[state][action] += learning_rate * (reward + discount_factor * Q[next_state][next_action] - Q[state][action])
# Print output to command line
print(f"Step {idx + 1}:")
print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")
print(f"Updated Q-value for state {state}, action {action}: {Q[state][action]}\n")
# Append results for visualization
rewards.append(reward)

return rewards

# Generate synthetic dataset
traffic_dataset = create_synthetic_traffic_dataset(100)

# Run SARSA using the dataset
rewards = sarsa_traffic_using_dataset(traffic_dataset)

# Plot cumulative rewards over episodes
def plot_rewards(rewards):
cumulative_rewards = np.cumsum(rewards)
plt.plot(cumulative_rewards)
plt.xlabel('Sample')
plt.ylabel('Cumulative Reward')
plt.title('SARSA Traffic Signal Control: Cumulative Reward Over Dataset')
plt.show()

# Plot the rewards
plot_rewards(rewards)

