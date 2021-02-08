import gym

# Choose the Cart-Pole environment from the OpenAI Gym


env = gym.make("Pong-v0")
obs = env.reset()

# Initialize the variables done (breaks loop) and total_rew (reward)
done = False
total_rew = 0

images = []

# Execution loop

win_points = 0
lose_points = 0

while not done:
    env.render()
    
    #Sample a random action from the environment
    ac = env.action_space.sample()

    
    #Obtain the new state, reward and whether the episode has terminated
    obs, rew, done, info = env.step(ac)
    obs_image = env.render(mode="rgb_array")
    
    if rew == -1:
        lose_points += 1
    if rew == 1:
        win_points += 1
    
    if win_points==3 or lose_points==3:
        done = True
    
    images.append(obs_image)

    # Accumulate the reward
    total_rew += rew

    
print(total_rew)

env.close()
