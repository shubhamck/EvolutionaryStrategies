import numpy as np
import random
import gym
import agent

# Declare GLobal Variables
NUM_ROLLOUTS = 50
NUM_EPOCHS = 1000
SIGMA = 0.1
ALPHA = 0.001



env = gym.make("CartPole-v0")

# Obs is 4x1 for cartpole
obs = env.reset()

#Reshape
obs = np.reshape(obs,(1,4))

#Initialize theta
theta = np.random.rand(4)

#Column vector
theta = theta.T
#print theta.shape

#Instantiate Agent
agent = agent.Agent(theta)

for i in range(NUM_EPOCHS):

	epsilons = []
	F = []
	
	for j in range(NUM_ROLLOUTS):
		obs = env.reset()
		obs = np.reshape(obs,(1,4))

		epsilon = np.random.normal(0,1)
		epsilons.append(epsilon)

		agent.pop_policy(epsilon, SIGMA)

		done = 0
		tot_reward = 0

		while not done:
		 	
		 	obs_new, reward, done, _ = env.step(agent.act(obs))
		 	tot_reward = tot_reward + reward

		 	obs = obs_new

		F.append(tot_reward)

	print "Episode : ",i," reward : ",sum(F)/float(len(F))

	# Reshape arrays for Updating
	F = np.reshape(np.asarray(F), (NUM_ROLLOUTS,1))

	F = (F - np.mean(F))/np.std(F)
	epsilons = np.reshape(np.asarray(epsilons), (NUM_ROLLOUTS,1))

	#Update thetas with F and epsilon
	agent.update(F,epsilons, ALPHA, NUM_ROLLOUTS, SIGMA)




