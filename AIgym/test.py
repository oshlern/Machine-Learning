# From Open AI Gym (https://gym.openai.com)
# import gym
# from gym import spaces
import numpy
print numpy.cross([1,2,0], [3,4,0])
# track = True
# space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
# x = space.sample

# env = gym.make('CartPole-v0')
#print(env.action_space
#print(env.observation_space)
# if track:
# import gym
# env = gym.make('CartPole-v0')
# env.monitor.start('/tmp/cartpole-experiment-2')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#
# env.monitor.close()
#
# gym.upload('/tmp/cartpole-experiment-1', api_key='YOUR_API_KEY')
# if track:
