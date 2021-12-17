import numpy as np
from utility_functions import *

def get_discrete_maximizer(phi, cdf, step_size):
	'''
		(phi-x)q(x) + int_0^x q(u)du maximzer is either at ceil(phi) or floor(phi), for discrete bids
	'''
	x1 = round_down(phi, step_size) #we assume the discrete bids are always on {0.01,...0.99,1.00}
	x2 = round_up(phi, step_size)
	if x1 > 1:
		x1 = 1.0
	if x2 > 1:
		x2 = 1.0
	val1 = (phi-x1)*cdf[int(x1/step_size)] + step_wise_integral(x1, cdf, step_size)
	val2 = (phi-x2)*cdf[int(x2/step_size)] + step_wise_integral(x2, cdf, step_size)
	return (x1,val1) if val1>=val2 else (x2,val2)

class BidFIDiscrete:
	def __init__(self, K=5, T=10**4, OB = None, ad_vals = None):
		'''
			K : absolute parity constraint
			T : number of ad rounds
			OB : OtherBidders object
			val_tab: is the value function i.e it gives the value for when [time,n_m-n_f+K,gender]
			bid_tab: gives the optimal bid for [time,n_m-n_f+K, gender]
		'''
		self.K = K
		self.T = T
		self.OB = OB
		# self.ad_vals[0], self.ad_vals[1] = ad_vals
		self.ad_vals = ad_vals

	def dp_bid_opt(self):
		'''
			val_tab[time slots remaining, state = (n_m-n_f+K), gender]
			bid_tab[timeslots remaining, state, gender]
		'''
		def pwin_term(gen,i,d):
			'''
				the terms associated with ( )*qwin
				i is time slot
				d is difference + K
			'''
			wd = d + (-1)**gen
			return self.ad_vals[gen] + p_ar[0]*val_t[i-1,wd,0] + p_ar[1]*val_t[i-1,wd,1] \
					-p_ar[0]*val_t[i-1,d,0] -p_ar[1]*val_t[i-1,d,1]

		def const_term(i,j):
			return p_ar[0]*val_t[i-1,j,0] + p_ar[1]*val_t[i-1,j,1]

		p_ar = [self.OB.prob_male, self.OB.prob_female]
		val_t = np.zeros((self.T, 2*self.K+1, 2)) #1st index is number of steps remaining till end, second index is N_m-N_f + K, 3rd is gender
		bid_t = np.zeros((self.T, 2*self.K+1, 2)) #non-stationary policy

		(x_m,val_m) = get_discrete_maximizer(self.ad_vals[0], self.OB.cdf_male, self.OB.step_size) #computes the integral see computing bid strategies section
		(x_f,val_f) = get_discrete_maximizer(self.ad_vals[1], self.OB.cdf_female, self.OB.step_size)

		val_t[0,:,0] = val_m
		val_t[0,:,1] = val_f
		val_t[0,2*self.K,0] = 0.0
		val_t[0,0,1] = 0.0

		bid_t[0,:,0] = x_m
		bid_t[0,:,1] = x_f
		bid_t[0,2*self.K,0] = 0.0
		bid_t[0,0,1] = 0.0

		for i in range(1, self.T):
			for j in range(2*self.K+1):
				if j != 0 and j != 2*self.K:
					(bid_t[i,j,0], val_m) = get_discrete_maximizer(pwin_term(0,i,j), self.OB.cdf_male, self.OB.step_size)
					(bid_t[i,j,1], val_f) = get_discrete_maximizer(pwin_term(1,i,j), self.OB.cdf_female,self.OB.step_size)
					val_t[i,j,0] = const_term(i,j) + val_m
					val_t[i,j,1] = const_term(i,j) + val_f
				elif j == 0: # i.e Nm-Nf = -K
					(bid_t[i,j,0], val_m) = get_discrete_maximizer(pwin_term(0,i,j), self.OB.cdf_male, self.OB.step_size)
					val_t[i,j,0] = const_term(i,j) + val_m
					bid_t[i,j,1] = 0.0
					val_t[i,j,1] = const_term(i,j)
				else: #i.e j==2*K
					(bid_t[i,j,1], val_f) = get_discrete_maximizer(pwin_term(1,i,j), self.OB.cdf_female,self.OB.step_size)
					val_t[i,j,1] = const_term(i,j) + val_f
					bid_t[i,j,0] = 0.0
					val_t[i,j,0] = const_term(i,j)
		self.val_tab_opt = val_t
		self.bid_tab_opt = bid_t

	def dp_bid_fixed(self, bid_mf):
		'''
			bid m when possible,
			bid f when possible.
		'''
		def pwin_term_bidfixed(gen,i,d):
			wd = d + (-1)**gen
			return self.ad_vals[gen] - bid_mf[gen] + p_ar[0]*val_t[i-1,wd,0] + p_ar[1]*val_t[i-1,wd,1] \
			- p_ar[0]*val_t[i-1,d,0] - p_ar[1]*val_t[i-1,d,1]
		def const_term(i,j):
			return p_ar[0]*val_t[i-1,j,0] + p_ar[1]*val_t[i-1,j,1]

		p_ar = [self.OB.prob_male, self.OB.prob_female]
		val_t = np.zeros((self.T, 2*self.K+1, 2))
		# index_m = int(round_down(bid_mf[0], self.OB.step_size) / self.OB.step_size) #used for finding pwin = cdf(floor(adval_m))
		# index_f = int(round_down(bid_mf[1], self.OB.step_size) / self.OB.step_size)
		index_m = int(bid_mf[0]/self.OB.step_size)
		index_f	= int(bid_mf[1]/self.OB.step_size)

		val_t[0,:,0] = (self.ad_vals[0] - bid_mf[0])*self.OB.cdf_male[index_m]+ step_wise_integral(bid_mf[0], self.OB.cdf_male, self.OB.step_size)
		val_t[0,:,1] = (self.ad_vals[1] - bid_mf[1])*self.OB.cdf_female[index_f]+ step_wise_integral(bid_mf[1], self.OB.cdf_female, self.OB.step_size)
		val_t[0,2*self.K,0] = 0.0
		val_t[0,0,1] = 0.0

		for i in range(1, self.T):
			for j in range(2*self.K+1):
				if j != 0 and j != 2*self.K:
					val_t[i,j,0] = pwin_term_bidfixed(0,i,j)*self.OB.cdf_male[index_m] + step_wise_integral(bid_mf[0],self.OB.cdf_male,self.OB.step_size) \
								   + const_term(i,j)
					val_t[i,j,1] = pwin_term_bidfixed(1,i,j)*self.OB.cdf_female[index_f]+ step_wise_integral(bid_mf[1], self.OB.cdf_female, self.OB.step_size) \
								  + const_term(i,j)
				elif j == 0:
					val_t[i,j,0] = pwin_term_bidfixed(0,i,j)*self.OB.cdf_male[index_m] + step_wise_integral(bid_mf[0],self.OB.cdf_male,self.OB.step_size) \
								   + const_term(i,j)
					val_t[i,j,1] = const_term(i,j)
				else: #i.e j==2*K
					val_t[i,j,0] = const_term(i,j)
					val_t[i,j,1] = pwin_term_bidfixed(1,i,j)*self.OB.cdf_female[index_f]+ step_wise_integral(bid_mf[1], self.OB.cdf_female, self.OB.step_size) \
								  + const_term(i,j)
		self.val_tab_det = val_t