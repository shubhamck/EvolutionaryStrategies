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
		#print "Epsilon : ",epsilon
		for i in range(self.theta.shape[0]):
			self.dtheta[i] = self.theta[i] + sigma*epsilon[i]


	def act(self, obs):
		"""
		Dot product of dtheta and obs such that -ve means left and +ve otherwise
		"""
		#print "theta shape", self.dtheta.shape
		#print "Obs shape: ", obs.shape
		action =  np.dot(obs,self.dtheta)

		if action < 0.5:
			return 0
		else:
			return 1



	def update(self, rewards, epsilons, alpha, rolls, sigma):

		"""
		Updates theta 
		"""
		#print self.theta
		#self.theta = self.theta.T
		#print self.theta

		#print  "Dot ",np.dot(epsilons.T,rewards)

		self.theta = self.theta + (alpha/(rolls*sigma))*np.dot(epsilons.T,rewards)
		#print self.theta

		#self.theta = self.theta.T
		#print self.theta
		




