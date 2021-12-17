from scipy.stats import betabinom
import numpy as np
from utility_functions import *
class OtherBidders:
	def __init__(self, nOthers = 9, prob_male = 0.5,step_size = 0.01):
		self.nOthers = nOthers
		self.prob_male = prob_male
		self.prob_female = 1.0 - prob_male
		self.cdf_male = None
		self.cdf_female = None
		self.step_size = step_size
		self.BB_n = None

	def set_cdf_BB(self, param_m = [0.9, 10], param_f=[0.9, 10]):
		'''
			cdf_male = betabinomial(n,alpha_m,beta_m)/n
			cdf_female = betabinomial(n,alpha_f,beta_f)/n
		'''
		self.BB_n = int (1.0 / self.step_size) #the n in Beta binomial
		self.x_bid = np.arange(0.0,1.0+self.step_size,self.step_size) #the support
		self.alpha_m, self.beta_m = param_m
		self.alpha_f, self.beta_f = param_f
		self.cdf_male = np.array([betabinom.cdf(i, self.BB_n, self.alpha_m, self.beta_m)**(self.nOthers) for i in range(self.BB_n+1)])
		self.cdf_female = np.array([betabinom.cdf(i, self.BB_n, self.alpha_f, self.beta_f)**(self.nOthers) for i in range(self.BB_n+1)])
		self.pmf_male = get_pmf(self.cdf_male)
		self.pmf_female = get_pmf(self.cdf_female)
		
	def draw_BBmax(self, gender):
		
		'''
			max of nOthers beta binomial
		'''
		alpha, beta = (self.alpha_m, self.beta_m) if gender==0 else (self.alpha_f, self.beta_f)
		return np.max(betabinom.rvs(self.BB_n, alpha, beta, size = self.nOthers)) / self.BB_n
		# if gender == 0:
		# 	return np.random.choice(self.x_bid, p=self.pmf_male)
		# else:
		# 	return np.random.choice(self.x_bid, p=self.pmf_female)