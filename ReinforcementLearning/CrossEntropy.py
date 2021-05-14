#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[3]:


import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


# ### NNet Params

# In[4]:


#Hidden layer
HIDDEN_SIZE=128
#Episodes per play
BATCH_SIZE=16
#Threshold
PERCENTILE=70


# ### Neural Network Definition

# In[6]:


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            # Notice there is no non-linearity function (yet)
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ### Episode and Step definition

# In[7]:


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# ### Function to batch iteration

# In[8]:


def iterate_batches(env, net, batch_size):
    """
    env: The environment (Env class instance from Gym)
    net: The defined neural network
    batch_size: count of episodes it should generate on every iteration
    """
    
    # Accumulates the current episode and its list of steps (EpisodeStep object)
    batch = []
    
    # Reset the environment
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    
    # To convert the nn output to a probability distribution of actions
    sm = nn.Softmax(dim=1)
    
    while True:
        # Current observation to a Pytorch tensor
        obs_v = torch.FloatTensor([obs])
        # Observation is passed to the NNet using Softmax for the Probability distribution
        act_probs_v = sm(net(obs_v))
        # Get the probability of the actions 
        act_probs = act_probs_v.data.numpy()[0]
        # Choose an action based on the probability distribution
        action = np.random.choice(len(act_probs), p=act_probs)
        # Apply that action on the environment
        next_obs, reward, is_done, _ = env.step(action)
        # Increment the reward
        episode_reward += reward
        # Add the step to the list 
        # - Warning; save the current observation, not the next_obs returned, affected with the action
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            # Append the episode 
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            # Reset the environment
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            
            if len(batch) == batch_size:
                # Return to the caller for processing using yield, so this function is a generator
                yield batch
                # And reset
                batch = []
        
        # Finally, update the current observation (State)
        obs = next_obs


# ### Filter those batche above the threshold defined

# In[9]:


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    # Filters the elite episodes, those with a reward over a percentile
    reward_bound = np.percentile(rewards, percentile)
    # Mean reward for monitoring purposes only
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    # For every filtered episode, save them and transform from Episodes into tensors and return a tuple:
    # - observations, actions, boundary of rewards and mean reward (last two for monitoring purposes)
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    # Return them
    return train_obs_v, train_act_v, reward_bound, reward_mean


# ### Execution

# In[12]:


if __name__ == "__main__":
    # Create environment
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    
    # Obs size
    obs_size = env.observation_space.shape[0]
    
    # Actions on this environment
    n_actions = env.action_space.n

    # Instantiate the NNet
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    
    #Objective function
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()


# In[ ]:




