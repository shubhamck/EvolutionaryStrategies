import numpy as np
import random
import gym
import agent
import matplotlib.pyplot as plt

# Declare GLobal Variables
NUM_ROLLOUTS = 50
NUM_EPOCHS = 500
SIGMA = 0.1
ALPHA = 0.001


np.random.seed(0)
env = gym.make("CartPole-v0")

# Obs is 4x1 for cartpole
obs = env.reset()

#Reshape
obs = np.reshape(obs,(1,4))

#Initialize theta
theta = np.random.rand(4)

#Column vector
theta = theta.T
#print theta.shape gives 4x1 vector

#Instantiate Agent
agent = agent.Agent(theta)

rewardList = []

for i in range(NUM_EPOCHS):
	#print "In ",i," epoch"

	#epsilons = []
	F = []
	epsilon = np.random.randn(NUM_ROLLOUTS, 4)
	for j in range(NUM_ROLLOUTS):
		#print "In ",j," Roll"
		obs = env.reset()
		obs = np.reshape(obs,(1,4))

		#epsilon = np.random.normal(0,1)
		#epsilon = np.random.randn(NUM_ROLLOUTS, 4)
		#epsilons.append(epsilon)

		agent.pop_policy(epsilon[j], SIGMA)

		done = 0
		tot_reward = 0

		while not done and tot_reward<501:
		 	
		 	obs_new, reward, done, _ = env.step(agent.act(obs))
		 	tot_reward = tot_reward + reward

		 	obs = obs_new

		F.append(tot_reward)

	rewardList.append(sum(F)/float(len(F)))
	print "Episode : ",i," reward : ",sum(F)/float(len(F))

	# Reshape arrays for Updating
	#F = np.reshape(np.asarray(F), (NUM_ROLLOUTS,1))
	F = np.asarray(F)

	F = (F - np.mean(F))/np.std(F)
	#epsilons = np.reshape(np.asarray(epsilons), (NUM_ROLLOUTS,1))

	#Update thetas with F and epsilon
	agent.update(F,epsilon, ALPHA, NUM_ROLLOUTS, SIGMA)
	#print "theta ",agent.theta

plt.plot(rewardList)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.show()




