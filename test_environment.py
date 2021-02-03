import gym

# Choose the Cart-Pole environment from the OpenAI Gym


env = gym.make("Pong-v0")
obs = env.reset()

# Initialize the variables done (breaks loop) and total_rew (reward)
done = False
total_rew = 0

images = []

# Execution loop
while not done:
    env.render()
    
    #Sample a random action from the environment
    ac = env.action_space.sample()

    
    #Obtain the new state, reward and whether the episode has terminated
    obs, rew, done, info = env.step(ac)
    print(rew)
    obs_image = env.render(mode="rgb_array")

    images.append(obs_image)

    # Accumulate the reward
    total_rew += rew
    
env.close()
