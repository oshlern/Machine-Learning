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
        value = numpy.dot(self.weights, observation) #+ self.bias
        value = observation[1]
        out = 1
        if value > 0:
            out = 2
        elif value < 0:
            out = 0
        # print value
        # time.sleep(0.5)
        return out

    def run(self, env, limit, render=False):
        observation, done = env.reset(), 0
        furthest = observation[0]
        for t in range(limit):
            action = self.action(observation)
            if render:
                env.render()
                print t, action
            observation, reward, done, info = env.step(action)
            if observation[0] > furthest:
                furthest = observation[0]
            if info != {}:
                print "info", info
            if done:
                if render:
                    print "COMPLETED!"
                    # time.sleep(1)
                break
        return furthest, done

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
        if reward:
            out += 1000
        return out/t

    def randAdjust(self, env, limit, render=False):
        furthest, reward = self.run(env, limit, render=render)
        # performance = self.value(observation, reward, t)
        temp = alg([self.bias + self.dw] + self.weights)
        tempFurthest, tempReward = temp.run(env, limit, render=render)
        change = self.dw * (tempFurthest-furthest)
        if render:
            print "ADJUSTING", change
            # time.sleep(1)
        self.bias += change
        for i in range(self.numWeights):
            newWeights = copy.deepcopy(self.weights)
            newWeights[i] += self.dw
            temp = alg([self.bias] + newWeights)
            tempFurthest, tempReward = temp.run(env, limit, render=False)
            change = self.dw * (tempFurthest-furthest)
            print "hi"
            if render:
                print i, change
                # time.sleep(1)
            self.weights[i] += change

    def randTrain(self, env, episodes, limit, render=False):
        for i_episode in range(episodes):
            self.printWeights()
            if render and i_episode%20 == 0:
                time.sleep(0.5)
            self.randAdjust(env, limit, render=render)
            # self.adjust(self.run(env, limit, render=render))
            print("Episode {} finished".format(i_episode))

    def test(self, env, episodes, limit, render=False):
        count = 0
        self.printWeights()
        for i_episode in range(episodes):
            furthest, reward = self.run(env, limit, render=render)
            if reward:
                count += 1
            print("Episode {} finished and {}".format(i_episode, reward))
        print "Completed {} out of {}, or {}%".format(count, episodes, count/episodes*100)
        return count/episodes*100

    def printWeights(self):
        print "Weights: ", [self.bias] + self.weights

env = gym.make('MountainCar-v0')
mc = alg()
# mc.printWeights()
# mc.randTrain(env, 500, 250, render=False)
# mc.randTrain(env, 50, 150, render=False)
# mc.printWeights()
# time.sleep(5)
# mc.run(env, 500, render=True)
# time.sleep(10)
# mc.test(env, 5, 120, render=True)

tests = []
n = 10
mc.test(env, 2000, 400, render=False)
# for i in range(n):
#     tests.append(mc.test(env, 1000, 1000, render=False))
#     print "\n\n\n\nITERATION " + str(i) + " COMPLETE"
# print sum(tests)/n
time.sleep(1)
