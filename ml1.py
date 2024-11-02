import numpy as np
import random
import matplotlib.pyplot as plt
# Define the grid size (city layout)
grid_size = 10
goal_positions = [(9, 9), (5, 5)]  # Example delivery locations
# Hyperparameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000
# Initialize the Q-table with zeros
q_table = np.zeros((grid_size, grid_size, 4))  # 4 possible actions: up, down, left, right

# Define actions
actions = ['up', 'down', 'left', 'right']
# Function to choose an action based on the epsilon-greedy strategy
def choose_action(state):
if random.uniform(0, 1) < epsilon:
return random.choice(actions)  # Explore
else:
return actions[np.argmax(q_table[state[0], state[1]])]  # Exploit
# Function to take an action and return the next state
def take_action(state, action):
x, y = state
if action == 'up' and x > 0:
x -= 1
elif action == 'down' and x < grid_size - 1:
x += 1
elif action == 'left' and y > 0:
y -= 1
elif action == 'right' and y < grid_size - 1:
y += 1
return (x, y)
# Function to calculate the reward based on the state
def get_reward(state):
if state in goal_positions:
return 10  # Reward for reaching a delivery location
return -1  # Penalty for each step taken
# Q-Learning algorithm
for episode in range(num_episodes):
state = (0, 0)  # Start position of the drone
while state not in goal_positions:
action = choose_action(state)
next_state = take_action(state, action)

# Get the reward
reward = get_reward(next_state)
# Update Q-value
current_q_value = q_table[state[0], state[1], actions.index(action)]
max_future_q = np.max(q_table[next_state[0], next_state[1]])  # Best Q-value for the next state
q_table[state[0], state[1], actions.index(action)] = current_q_value + learning_rate * (reward + discount_factor * max_future_q - current_q_value)
# Move to the next state
state = next_state
# Display the Q-table
print("Trained Q-Table:")
print(q_table)
# To simulate real-time decision making
def real_time_decision_making(start_state):
current_state = start_state
path_taken = [current_state]
while current_state not in goal_positions:
action = actions[np.argmax(q_table[current_state[0], current_state[1]])]
current_state = take_action(current_state, action)
path_taken.append(current_state)
return path_taken
# Example of a real-time decision from the start position
path = real_time_decision_making((0, 0))
print("Path taken by the drone to the goal:", path)
# Visualization
def visualize_path(path):
# Create a grid
grid = np.zeros((grid_size, grid_size))
# Mark the goal positions
for goal in goal_positions:
grid[goal] = 2  # Mark goals with 2
# Mark the path taken by the agent
for position in path:
grid[position] = 1  # Mark path with 1
# Create the plot
plt.imshow(grid, cmap='Greys', origin='upper')
plt.colorbar(ticks=[0, 1, 2], label='Grid Value')
plt.clim(-0.5, 2.5)  # Set color limits for better visibility
plt.title("Q-Learning Path Visualization")
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))
plt.grid(False)
# Show the grid
plt.show()
# Visualize the path taken by the agent
visualize_path(path)
