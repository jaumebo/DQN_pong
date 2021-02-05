import os
from datetime import datetime

import torch

from src.dqn_funcs import *
from src.visualize import *
from src.visualize import test_gif

from torch.utils.tensorboard import SummaryWriter
import numpy as np

#Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using GPU:', ['no', 'yes'][int(torch.cuda.is_available())])

#Set saving directories
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists('gifs'):
    os.makedirs('gifs')

if not os.path.exists('logs'):
    os.makedirs('logs')   

t = datetime.now()
current_time = t.strftime("%Y_%m_%d-%H_%M_%S")

os.makedirs('gifs/' + current_time)
os.makedirs('checkpoints/' + current_time)
os.makedirs('checkpoints/' + current_time + '/last/')
os.makedirs('checkpoints/' + current_time + '/best/')
writer = SummaryWriter('logs/' + current_time)


#Hyperparameters
env_name = 'Pong-ram-v0'
gamma = 0.99  # discount factor
seed = 123  # random seed
log_interval = 50  # controls how often we log progress, in episodes
model_save_interval = 100 # controls how often we save the model at checkpoints (in episodes)
num_episodes = 500000  # number of episodes to train on
batch_size = 32  # batch size for optimization
lr = 1e-4  # learning rate
eps_start = 1.0  # initial value for epsilon (in epsilon-greedy)
eps_end = 0.02  # final value for epsilon (in epsilon-greedy)
eps_decay = 1000000  # length of epsilon decay, previous 1000000
target_update = 100  # how often to update target net, in env steps, previous 1000, 100
memory_size = 500000 # how many steps we keep in memory, previous 100000, 500000

# Create environment
env = gym.make(env_name)

# Fix random seed (for reproducibility)
env.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Get number of actions from gym action space
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n

# Create Policy Networks
policy_net = DQN_ram(n_inputs, n_actions).to(device)
target_net = DQN_ram(n_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(memory_size)

step_count = 0
ep_rew_history = []
i_episode = 0
ep_reward = -float('inf')
sum_reward = 0
avg_reward = 0
best_avg_reward = -100

while i_episode < num_episodes:
    # Initialize the environment and state
    state = env.reset()
    done = False
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    episode_reward = 0
    win_points = 0
    lose_points = 0

    while not done:
        # Select an action
        eps_greedy_threshold = compute_eps_threshold(step_count, eps_start, eps_end, eps_decay)
        action = select_action(policy_net, state, eps_greedy_threshold, n_actions, device)

        # Perform action in env
        next_state, reward, done, _ = env.step(action.item())

        if reward == -1:
            lose_points += 1
        elif reward == 1:
            win_points += 1
        
        if win_points==3 or lose_points==3:
            done = True

        # Bookkeeping
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        reward = reward_shaper(reward, done)
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        step_count += 1

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        stop = train(policy_net, target_net, optimizer, memory, batch_size, gamma, device) 

        # Update the target network, copying all weights and biases in DQN
        if step_count % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    i_episode += 1
    sum_reward += episode_reward
    avg_reward = sum_reward/i_episode
    writer.add_scalar('Rewards', episode_reward, i_episode)
    writer.add_scalar('Avg rewards', avg_reward, i_episode)

    # Evaluate greedy policy
    if i_episode % log_interval == 0 or i_episode >= num_episodes:            
        ep_reward = test_gif(device, policy_net, 'gifs/' + current_time + '/test_gif_ep' + str(i_episode) + '.gif',env_name)
        ep_rew_history.append((i_episode, ep_reward))
        print('Episode {}\tSteps: {:.2f}k''\tAvg reward: {:.2f}''\tEval reward: {:.2f}'.format(i_episode, step_count/1000., avg_reward, ep_reward))

    if i_episode % model_save_interval == 0 or i_episode >= num_episodes:
        torch.save(policy_net.state_dict(), 'checkpoints/' + current_time  + '/last/last_dqn-{}.pt'.format(env_name))
        print("Saved model checkpoint")
    
    if avg_reward>best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(policy_net.state_dict(), 'checkpoints/' + current_time  + '/best/best_dqn-{}.pt'.format(env_name))
        print("Saved model checkpoint")

print("Finished training! Eval reward: {:.2f}".format(ep_reward))


