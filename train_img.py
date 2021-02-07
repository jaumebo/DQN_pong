import os
from datetime import datetime

import torch

from src.dqn_funcs import *
from src.visualize import *
from src.visualize import test_agent

from torch.utils.tensorboard import SummaryWriter
import numpy as np

#Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using GPU:', ['no', 'yes'][int(torch.cuda.is_available())])
print('Device: ')
print(device)

#Set saving directories
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists('gifs'):
    os.makedirs('gifs')

if not os.path.exists('logs'):
    os.makedirs('logs')   

t = datetime.now()
current_time = t.strftime("%Y_%m_%d-%H_%M_%S")
name = current_time + "_img"

os.makedirs('gifs/' + name)
os.makedirs('checkpoints/' + name)
os.makedirs('checkpoints/' + name + '/last/')
os.makedirs('checkpoints/' + name + '/best/')
writer = SummaryWriter('logs/' + name)


#Hyperparameters
env_name = 'Pong-v0'
gamma = 0.99  # discount factor
seed = 123  # random seed
log_interval = 100  # controls how often we log progress, in episodes
gif_interval = 400
model_save_interval = 400 # controls how often we save the model at checkpoints (in episodes)
num_episodes = 2000000  # number of episodes to train on
batch_size = 32  # batch size for optimization
lr = 1e-4  # learning rate
target_update = 10000  # how often to update target net, in env steps
memory_size = 1000000 # how many steps we keep in memory
memory_start_size = 5000
skip_frames = 8 # frames skip from the game, this helps the agent to faster see more situations of the game

# Create environment
env = gym.make(env_name)

# Fix random seed (for reproducibility)
env.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Get number of actions from gym action space
n_actions = env.action_space.n

# Create Policy Networks
policy_net = DQN_img(3, n_actions)
policy_net.to(device)
target_net = DQN_img(3, n_actions)
target_net.to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(memory_size)
eps_scheduler = epsilon_scheduler(memory_start_capacity=memory_start_size)

step_count = 0
i_episode = 0
ep_reward = -float('inf')
sum_reward = 0
sum_reward_window = 0
best_avg_reward = -100
logged_start = False

print("Preliminary example:")
ep_reward = test_agent(device, policy_net, 'gifs/' + name + '/test_gif_ep' + str(i_episode) + '.gif',env_name,True)
print("First ep_reward: " + str(ep_reward))

while i_episode < num_episodes:
    # Initialize the environment and state
    state = env.reset()
    done = False
    state = get_state_img(state)
    episode_reward = 0

    while not done:
        # Select an action
        eps_greedy_threshold = eps_scheduler.get_epsilon(step_count)
        action = select_action(policy_net, state, eps_greedy_threshold, n_actions, device)

        # Perform action in env for "skip_frames" number of times
        for _ in range(skip_frames):
            next_state, reward, done, _ = env.step(action.item())            

            # Bookkeeping
            next_state = get_state_img(next_state)
            reward = reward_shaper(reward, done)
            episode_reward += reward
            reward = torch.tensor([reward], device=device)
            step_count += 1

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                break

        if step_count<memory_start_size:
            continue

        # Perform one step of the optimization (on the policy network)
        if not logged_start:
            print("Start training the network")
            logged_start = True
        
        stop = train(policy_net, target_net, optimizer, memory, batch_size, gamma, device) 

        # Update the target network, copying all weights and biases in DQN
        if step_count % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if step_count<memory_start_size:
        continue

    i_episode += 1
    sum_reward += episode_reward
    sum_reward_window += episode_reward
    avg_reward = sum_reward/i_episode
    writer.add_scalar('Rewards', episode_reward, i_episode)
    writer.add_scalar('Avg rewards', avg_reward, i_episode)
    writer.add_scalar('Epsilon value', eps_greedy_threshold, i_episode)
    

    # Evaluate greedy policy
    if i_episode % log_interval == 0 or i_episode >= num_episodes:

        if i_episode % gif_interval == 0:
            savegif=True
        else:
            savegif=False
        
        ep_reward = test_agent(device, policy_net, 'gifs/' + name + '/test_gif_ep' + str(i_episode) + '.gif',env_name,savegif)
        writer.add_scalar('Validation value', ep_reward, i_episode)
        print('Episode {}\tSteps: {:.2f}k''\tAvg reward: {:.4f}''\tEval reward: {:.2f}''\tEpsilon value: {:.6f}'.format(i_episode, step_count/1000., avg_reward, ep_reward, eps_greedy_threshold))

    if i_episode % model_save_interval == 0 or i_episode >= num_episodes:
        torch.save(policy_net.state_dict(), 'checkpoints/' + name  + '/last/last_dqn-{}.pt'.format(env_name))
        print("Saved model checkpoint")
    
    if (i_episode % model_save_interval == 0 or i_episode >= num_episodes) and avg_reward_window_value>best_avg_reward and i_episode>2000:
        best_avg_reward = avg_reward
        torch.save(policy_net.state_dict(), 'checkpoints/' + name  + '/best/best_dqn-{}.pt'.format(env_name))
        print("Saved best model checkpoint")

print("Finished training! Eval reward: {:.2f}".format(ep_reward))


