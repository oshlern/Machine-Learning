from __future__ import division
# From Open AI Gym (https://gym.openai.com)
import gym, time, numpy, copy, random
# This is my attempt at creating a structure for a reinforcement algorithm that could be implemented in different tasks.
# It is currently fit to MountainCar. It is currently just a perceptron.
# The algorithm uses the OpenAI env to render, act, and observe.
# It evaluates its current state (value), and then tweaks its constants minutely and reevaluates to calculate the derivative of error
# Using the derivative, it updates the weights (and repeats: randTrain).
# It can also evaluate itself with test.
class alg:
    def __init__(self, weights=[0, 0, 0]):
        # TODO: Tie algorithm to environment? Probably
        self.bias = weights[0]
        self.weights = weights[1:]
        self.numWeights = len(weights)

    def action(self, observation):
        value = numpy.dot(self.weights, observation) + self.bias
        out = 0
        if value > 0:
            out = 2
        elif value < 0:
            out = 1
        return out

    def run(self, env, limit, render=False, evaluate=False):
        # TODO: observe general run, not just end?
        observation, reward, performance = env.reset(), 0, 0
        for t in range(limit):
            action = self.action(observation)
            if render:
                env.render()
                # print action
            observation, reward, done, info = env.step(action)
            if evaluate:
                performance += observation
            if info != {}:
                print "info", info
            if done:
                return observation, reward, done, t, performance
        return observation, reward, done, limit, performance

    def value(self, observation, reward, done, t, evaluation=0):
        out = evaluation + sum([abs(o) for o in observation]
        if int(reward):
            out += 1000
        return out/t

    def randAdjust(self, env, limit, render=False):
        # Tries to approximate derivatives and adjust porportionally
        observation, reward, done, t, performance = self.run(env, limit, render=render)
        performance = self.value(observation, reward, t)

        temp = alg([self.bias + 0.01] + self.weights)
        tempO, tempR, tempT = temp.run(env, limit, render=render)
        change = 0.01 * (temp.value(tempO, tempR, tempT)-performance)
        print "ADJUSTING", change
        self.bias -= change
        for i in range(self.numWeights-1):
            newWeights = copy.deepcopy(self.weights)
            newWeights[i] += 0.01
            temp = alg([self.bias] + newWeights)
            tempO, tempR, tempT = temp.run(env, limit, render=render)
            change = 0.01 * (temp.value(tempO, tempR, tempT)-performance)
            print i, change
            self.weights[i] -= change

    def randTrain(self, env, episodes, limit, render=False):
        for i_episode in range(episodes):
            print "weights, ", self.weights
            self.randAdjust(env, limit, render=render)
            # self.adjust(self.run(env, limit, render=render))
            print("Episode {} finished".format(i_episode))

    def test(self, env, episodes, limit, render=False):
        times = []
        print self.bias, self.weights
        for i_episode in range(episodes):
            observation, reward, t = self.run(env, limit, render=render)
            print observation
            if t != limit:
                times.append(t)
            print("Episode {} finished after {} timesteps".format(i_episode, t))
        print "Completed {} out of {}, or {}%".format(len(times), episodes, len(times)/episodes*100)
        print "average time: {}".format(sum(times)/(len(times)+0.0001))
        return len(times)/episodes*100

    def printWeights(self):
        print self.allWeights

env = gym.make('MountainCar-v0')
mc = alg()
# mc.printWeights()
mc.randTrain(env, 100, 500)
# mc.printWeights()
# mc.run(env, 500, render=True)
# time.sleep(10)
mc.test(env, 25, 120, render=True)

tests = []
n = 10
# for i in range(n):
#     tests.append(mc.test(env, 1000, 1000, render=False))
#     print "\n\n\n\nITERATION " + str(i) + " COMPLETE"
# print sum(tests)/n
time.sleep(1)




# def adjust(self, observation, reward, t):
#     v = -(reward-1.1)/t # = self.value(observation, reward, t)
#     self.weights = [w + o/reward for w,o in zip(self.weights, observation)]
#     self.bias += 1/(observation[0] + observation[1] + 0.1)
#
# def train(self, env, episodes, limit, render=False):
#     for i_episode in range(episodes):
#         observation, reward, t = self.run(env, limit, render=render)
#         self.adjust(observation, reward, t)
#         if render:
#             print self.bias, self.weights
#             print("Episode {} finished after {} timesteps".format(i_episode, t))
