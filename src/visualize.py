import gym
import imageio
import torch

from src import dqn_funcs

def test_agent(device,policy_net=None,path='test.gif',env_name="Pong-ram-v0",savegif=True):

    env = gym.make(env_name)

    obs = env.reset()

    # Initialize the variables done (breaks loop) and total_rew (reward)
    done = False
    total_rew = 0

    images = []

    # Execution loop
    while not done:
        #env.render(mode='rgb_array')
        
        if policy_net is None:
            #Sample a random action from the environment
            ac = env.action_space.sample()
        else:
            #state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            state = dqn_funcs.get_state_img(obs).to(device)
            ac = dqn_funcs.select_action(policy_net, state, 0., 1,device).item()
        
        #Obtain the new state, reward and whether the episode has terminated
        obs, rew, done, info = env.step(ac)
        obs_image = env.render(mode="rgb_array")

        images.append(obs_image)

        # Accumulate the reward
        total_rew += rew
        
    env.close()

    # Save to GIF
    if savegif:
        imageio.mimsave(path, images, fps=60)

    return total_rew


