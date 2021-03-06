from __future__ import division
# From Open AI Gym (https://gym.openai.com)
import gym, time, numpy, copy, random
# TODO: test a bunch for performance, not just once (environment changes)
# TODO: Automatically normalize observations
class alg:
    dw = 0.01
    learningRate = 10
    def __init__(self, weights=[random.random()*0.1, random.random(), random.random()]):
        self.bias = weights[0]
        self.weights = weights[1:]
        self.numWeights = len(weights)

    def action(self, observation):
        # observation[1] *= 10
        value = numpy.dot(self.weights, observation) + self.bias/5000
        # print numpy.dot(self.weights, observation)
        out = 1
        if value > 0:
            out = 2
        elif value < 0:
            out = 0
        return out

    def run(self, env, limit, render=False):
        observation, done = env.reset(), 0
        furthest = observation[0]
        for t in range(limit):
            action = self.action(observation)
            if render:
                env.render()
                # if t%30== 0:
                print t, action, observation
                # time.sleep
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

    def averagePerformance(self, env, limit, trials):
        avg = 0
        for i in range(trials):
            furthest, done = self.run(env, limit)
            avg += furthest
        return avg/trials

    def randAdjust(self, env, limit, trials, index, episodes):
        performance = self.averagePerformance(env, limit, trials)
        ws = [self.bias] + self.weights
        for i in range(self.numWeights):
            newWeights = copy.deepcopy(ws)
            newWeights[i] += self.dw
            temp = alg(newWeights)
            change = temp.averagePerformance(env, limit, trials) - performance
            adjust = (episodes - index)/episodes * self.learningRate * change/self.dw
            if i == 0:
                self.bias += adjust
            else:
                self.weights[i-1] += adjust

    def randTrain(self, env, episodes, limit, trials=25, render=False):
        for i_episode in range(episodes):
            if render:# and i_episode%20 == 0:
                self.printWeights()
                # if i_episode % 20 == 10:
                #     self.run(env, 150, render=True)
            self.randAdjust(env, limit, trials, i_episode, episodes)
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
mc.randTrain(env, 50, 500, render=True)

# mc.randTrain(env, 50, 150, render=False)
mc.printWeights()
time.sleep(3)
# mc.weights[1] *= 1000
mc.run(env, 200, render=True)
mc.test(env, 200, 400, render=False)
time.sleep(1)
