import math
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.logged_full = False

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        elif not self.logged_full:
            print("Memory full! Start popping episodes.")
            self.logged_full = True
        self.memory[self.position] = Transition(*args)
        # Update the pointer to the next position in the replay memory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_img(nn.Module):
    def __init__(self, in_channels, outputs):
        super(DQN_img, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(64,1024,kernel_size=7,stride=1)
        self.fc1 = nn.Linear(163840,outputs)

    def forward(self, x):
        x = x.float() / 255
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

class DQN_ram(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN_ram, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

def get_state_img(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

class epsilon_scheduler():
    
    def __init__(self, 
                eps_start=1, 
                eps_mid_end=0.1, 
                eps_end=0.01, 
                exploration_frames=1000000, 
                smoothed_final_frames=25000000,
                memory_start_capacity=5000):
        
        self.eps_start = eps_start
        self.eps_mid_end = eps_mid_end
        self.eps_end = eps_end
        self.exploration_frames = exploration_frames
        self.smoothed_final_frames = smoothed_final_frames
        self.memory_start_capacity = memory_start_capacity

        self.slope1 = -(self.eps_start - self.eps_mid_end)/self.exploration_frames
        self.slope2 = -(self.eps_mid_end - self.eps_end)/(self.smoothed_final_frames - self.exploration_frames - self.memory_start_capacity)

    def get_epsilon(self,step):

        corrected_step = max(0,step - self.memory_start_capacity)

        if step<=(self.exploration_frames+self.memory_start_capacity):
            epsilon = self.eps_start + (corrected_step*self.slope1)
        elif step>(self.exploration_frames+self.memory_start_capacity) and step<(self.exploration_frames+self.smoothed_final_frames+self.memory_start_capacity):
            epsilon = self.eps_mid_end + (corrected_step*self.slope2)
        else:
            epsilon = self.eps_end
        
        return epsilon


def select_action(policy, state, eps_greedy_threshold, n_actions, device):
    if random.random() > eps_greedy_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state_cuda = state.to(device)
            action = policy(state_cuda).max(1)[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action

def train(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    transitions = memory.sample(batch_size)
    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=device, 
        dtype=torch.bool)
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t) for all a, then we select 
    # the columns of actions taken. These are the actions which would've been 
    # taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Q(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # Note the call to detach() on Q(s_{t+1}), which prevents gradient flow
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute targets for Q values: y_t = r_t + max(Q_{t+1})
    expected_state_action_values = reward_batch + (next_state_values * gamma)

    # Compute Huber loss between predicted Q values and targets y
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)).to(device)

    # Take an SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return True
    

def reward_shaper(rew, done):
    # Function ready to shape the reward if needed
    return rew


