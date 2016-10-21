from __future__ import division
# From Open AI Gym (https://gym.openai.com)
import gym, time, numpy, copy


env = gym.make('CartPole-v0')
x = {'p': 0, 'd': 0, 'i': 0}
a = {'p': 0, 'd': 0, 'i': 0}
constants = [0]*6 #{'xp': 0, 'xd': 0, 'xi': 0, 'ap': 0, 'ad': 0, 'ai': 0}
lastConstants = constants
lastResult = 0
attempt = 0
for i_episode in range(3000):
    observation = env.reset()
    xi, ai = 0, 0
    # cs = numpy.divide(constants, i_episode)
    for t in range(10000):
        # env.render()
        # print(observation)
        action = 0
        if attempt > 0:
            action = 1
        observation, reward, done, info = env.step(action)
        # x['p'], x['d'] = observation[0], observation[1]
        xi += observation[0]
        # a['p'], a['d'] = observation[2], observation[3]
        ai += observation[2]
        attempt = sum(numpy.multiply(constants[:4], observation)) - constants[4]*xi - constants[5]*xi#* a['p'] + 5 * a['d'] + x['p'] + 2*x['d']
        # time.sleep(0.0001)
        if info != {}:
            print "info", info
        if done:
            # print "ints", [xi, ai]
            # if t > lastResult:
                # numpy.divide(observation, (t**4+100)/100000)
                # if constants[3]**2 <= 0.02:
                    # print observation[2], "SUIDGAGIU"
                # lastConstants = copy.deepcopy(constants)
            divisor = (t**10+100**5)/100**10
            # if t<400:
            constants[2] += observation[2]**2/divisor#*100
            constants[3] += (observation[3]**4 + observation[2]**2)/divisor#*7
            # else:
            constants[0] += observation[0]**2/divisor#*11
            constants[1] += observation[1]**2/divisor#/10
            constants[4] += xi/10000
            constants[5] += ai/10000
                # lastResult = t
                # print "\n______NEW", t
                # print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            # else:
                # constants = copy.deepcopy(lastConstants)
                # print t, lastResult
                # time.sleep(0.1)
            # print "constants", constants
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
env.reset()
# env.render()
for t in range(10000):
    env.render()
    if t%100 == 0:
        print t
    # print(observation)
    action = 0
    if attempt > 0:
        action = 1
    observation, reward, done, info = env.step(action)
    # x['p'], x['d'] = observation[0], observation[1]
    xi += observation[0]
    # a['p'], a['d'] = observation[2], observation[3]
    ai += observation[2]
    attempt = sum(numpy.multiply(constants[:4], observation)) - constants[4]*xi - constants[5]*xi#* a['p'] + 5 * a['d'] + x['p'] + 2*x['d']
    time.sleep(0.01)
    if done:
        print t
        break
time.sleep(1)
