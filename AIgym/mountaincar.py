from __future__ import division
# From Open AI Gym (https://gym.openai.com)
import gym, time, numpy, copy, random

class alg:
    dw = 0.01

    def __init__(self, weights=[0, 0, 0]):
        self.bias = weights[0]
        self.weights = weights[1:]
        # self.allWeights = weights
        self.numWeights = len(weights) - 1

    def action(self, observation):
        value = numpy.dot(self.weights, observation) + self.bias
        out = 0
        if value > 0:
            out = 2
        elif value < 0:
            out = 1
        return out

    def run(self, env, limit, render=False, evaluate=False):
        observation, reward = env.reset(), 0
        furthest = observation[0]
        for t in range(limit):
            action = self.action(observation)
            if render:
                env.render()
                # print action
            observation, reward, done, info = env.step(action)
            if observation[0] > furthest:
                furthest = observation[0]
            if info != {}:
                print "info", info
            if done:
                return observation, reward, t
        return furthest, reward

    def adjust(self, observation, reward, t):
        v = -(reward-1.1)/t # = self.value(observation, reward, t)
        self.weights = [w + o/reward for w,o in zip(self.weights, observation)]
        self.bias += 1/(observation[0] + observation[1] + 0.1)

    def train(self, env, episodes, limit, render=False):
        for i_episode in range(episodes):
            observation, reward, t = self.run(env, limit, render=render)
            self.adjust(observation, reward, t)
            if render:
                print self.bias, self.weights
                print("Episode {} finished after {} timesteps".format(i_episode, t))

    def value(self, observation, reward, t):
        out = abs(observation[0]) + abs(observation[1])
        if int(reward):
            out += 1000
        return out/t

    def randAdjust(self, env, limit, render=False):
        furthest, reward = self.run(env, limit, render=render)
        # performance = self.value(observation, reward, t)
        temp = alg([self.bias + self.dw] + self.weights)
        tempFurthest, tempReward = temp.run(env, limit, render=render)
        change = self.dw * (tempFurthest-furthest)
        print "ADJUSTING", change
        self.bias += change
        for i in range(self.numWeights):
            newWeights = copy.deepcopy(self.weights)
            newWeights[i] += self.dw
            temp = alg([self.bias] + newWeights)
            tempFurthest, tempReward = temp.run(env, limit, render=render)
            change = self.dw * (tempFurthest-performance)
            print i, change
            self.weights[i] -= change

    def randTrain(self, env, episodes, limit, render=False):
        for i_episode in range(episodes):
            self.printWeights()
            self.randAdjust(env, limit, render=render)
            # self.adjust(self.run(env, limit, render=render))
            print("Episode {} finished".format(i_episode))

    def test(self, env, episodes, limit, render=False):
        times = []
        self.printWeights()
        for i_episode in range(episodes):
            furthest, reward = self.run(env, limit, render=render)
            print observation
            if t != limit:
                times.append(t)
            print("Episode {} finished after {} timesteps".format(i_episode, t))
        print "Completed {} out of {}, or {}%".format(len(times), episodes, len(times)/episodes*100)
        print "average time: {}".format(sum(times)/(len(times)+0.0001))
        return len(times)/episodes*100

    def printWeights(self):
        print "Weights: ", [self.bias] + self.weights

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
