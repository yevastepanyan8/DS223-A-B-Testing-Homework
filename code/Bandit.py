"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from abc import ABC, abstractmethod

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization:

    def plot1(self, rewards_egreedy, rewards_ts):
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(rewards_egreedy), label="Epsilon-Greedy")
        plt.plot(np.cumsum(rewards_ts), label="Thompson Sampling")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.savefig('plot1_rewards.png')
        plt.show(block=False)
        plt.pause(10)
        plt.close()

    def plot2(self, regrets_egreedy, regrets_ts):
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(regrets_egreedy), label="Epsilon-Greedy")
        plt.plot(np.cumsum(regrets_ts), label="Thompson Sampling")
        plt.title("Cumulative Regret")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.savefig('plot2_regrets.png')
        plt.show(block=False)
        plt.pause(10)
        plt.close()

#--------------------------------------#

class EpsilonGreedy(Bandit):

    def __init__(self, p, n_trials=20000, epsilon=0.1, decay=True):
        self.p = p  # true probabilities
        self.n_trials = n_trials
        self.epsilon = epsilon
        self.decay = decay
        self.n_bandits = len(p)
        self.estimates = np.zeros(self.n_bandits)
        self.N = np.zeros(self.n_bandits)
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return f"EpsilonGreedy(eps={self.epsilon}, trials={self.n_trials})"

    def pull(self, bandit):
        return np.random.random() < self.p[bandit]

    def update(self, bandit, reward):
        self.N[bandit] += 1
        self.estimates[bandit] += (reward - self.estimates[bandit]) / self.N[bandit]

    def experiment(self):
        optimal_reward = max(self.p)
        for t in range(1, self.n_trials + 1):
            eps = self.epsilon / t if self.decay else self.epsilon
            if np.random.random() < eps:
                bandit = np.random.randint(self.n_bandits)
            else:
                bandit = np.argmax(self.estimates)

            reward = self.pull(bandit)
            self.update(bandit, reward)
            self.rewards.append(reward)
            regret = optimal_reward - self.p[bandit]
            self.regrets.append(regret)
        logger.info("Epsilon-Greedy experiment finished.")

    def report(self):
        total_reward = np.sum(self.rewards)
        total_regret = np.sum(self.regrets)
        print(f"[Epsilon-Greedy] Cumulative Reward: {total_reward:.2f}")
        print(f"[Epsilon-Greedy] Cumulative Regret: {total_regret:.2f}")
        data = pd.DataFrame({
            "Bandit": np.argmax(self.estimates),
            "Reward": self.rewards,
            "Algorithm": "EpsilonGreedy"
        })
        data.to_csv("epsilon_greedy_rewards.csv", index=False)
        return data

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p, n_trials=20000, precision=1):
        self.p = p
        self.n_trials = n_trials
        self.precision = precision
        self.n_bandits = len(p)
        self.successes = np.zeros(self.n_bandits)
        self.failures = np.zeros(self.n_bandits)
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return f"ThompsonSampling(precision={self.precision}, trials={self.n_trials})"

    def pull(self, bandit):
        return np.random.random() < self.p[bandit]

    def update(self, bandit, reward):
        if reward == 1:
            self.successes[bandit] += 1
        else:
            self.failures[bandit] += 1

    def experiment(self):
        optimal_reward = max(self.p)
        for _ in range(self.n_trials):
            sampled_theta = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1)
                             for i in range(self.n_bandits)]
            bandit = np.argmax(sampled_theta)
            reward = self.pull(bandit)
            self.update(bandit, reward)
            self.rewards.append(reward)
            regret = optimal_reward - self.p[bandit]
            self.regrets.append(regret)
        logger.info("Thompson Sampling experiment finished.")

    def report(self):
        total_reward = np.sum(self.rewards)
        total_regret = np.sum(self.regrets)
        print(f"[Thompson Sampling] Cumulative Reward: {total_reward:.2f}")
        print(f"[Thompson Sampling] Cumulative Regret: {total_regret:.2f}")
        data = pd.DataFrame({
            "Bandit": np.argmax(self.p),
            "Reward": self.rewards,
            "Algorithm": "ThompsonSampling"
        })
        data.to_csv("thompson_rewards.csv", index=False)
        return data


def comparison():
    Bandit_Reward = [1, 2, 3, 4]
    trials = 20000

    egreedy = EpsilonGreedy(Bandit_Reward, n_trials=trials)
    ts = ThompsonSampling(Bandit_Reward, n_trials=trials)

    egreedy.experiment()
    ts.experiment()

    eg_data = egreedy.report()
    ts_data = ts.report()

    viz = Visualization()
    viz.plot1(egreedy.rewards, ts.rewards)
    viz.plot2(egreedy.regrets, ts.regrets)

    combined = pd.concat([eg_data, ts_data])
    combined.to_csv("bandit_results.csv", index=False)

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")