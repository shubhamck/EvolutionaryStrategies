import numpy as np
import random

class Agent:

	def __init__(self,theta):
		self.theta = theta
		self.dtheta = theta

	def pop_policy(self, epsilon, sigma):
		"""
		epsilon = vector sampled from Normal Distribution with mean zero and std 1
		sigma = scalar : Std of noise
		"""
		for i in range(self.theta.shape[0]):
			self.dtheta[i] = self.theta[i] + sigma*epsilon


	def act(self, obs):
		"""
		Dot product of dtheta and obs such that -ve means left and +ve otherwise
		"""
		#print "theta shape", self.dtheta.shape
		#print "Obs shape: ", obs.shape
		action =  np.dot(obs,self.dtheta)

		if action > 0:
			return 0
		else:
			return 1



	def update(self, rewards, epsilons, alpha, rolls, sigma):

		"""
		Updates theta 
		"""

		for i in range(self.theta.shape[0]):
			self.theta[i] = self.theta[i] + (alpha/(rolls*sigma))*np.dot(rewards.T, epsilons)




