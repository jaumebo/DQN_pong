import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # TODO: Update the pointer to the next position in the replay memory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
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

def compute_eps_threshold(step, eps_start, eps_end, eps_decay):
  return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)


def select_action(policy, state, eps_greedy_threshold, n_actions, device):
    if random.random() > eps_greedy_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy(state).max(1)[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action

def train(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    if len(memory) < batch_size:
        return False
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
        [s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
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
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute targets for Q values: y_t = r_t + max(Q_{t+1})
    expected_state_action_values = reward_batch + (next_state_values * gamma)

    # Compute Huber loss between predicted Q values and targets y
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1))

    # Take an SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return True
    

def reward_shaper(rew, done):
    # The default reward is +1 for any (s,a) pair. We can make the task easier
    # by giving a negative reward (e.g. -10) when the pole falls.
    # You can try modifying/removing this reward shaping and see what happens.
    # This is a clear example of the importance of reward in RL.
    return rew


