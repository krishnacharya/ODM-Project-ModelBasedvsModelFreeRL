import numpy as np
from utility_functions import *
# from random import randint

class PSRLTypeAlgos:
	'''
		Discrete action PSRL type algorithms
	'''
	def __init__(self, fi_obj=None):
		'''
			fi_obj: Object of class BidFairFullInfo, we use its methods e.g dp
		'''
		self.fi_obj = fi_obj

	def valiter(self, pwin_est, epsilon_ev = 1e-3, gam = 1-1e-6):
		u_old = np.random.uniform(low = 0.0, high = 1.0, size = (2, 2 * self.fi_obj.K + 1))
		u_new = np.random.uniform(low = 0.0, high = 1.0, size = (2, 2 * self.fi_obj.K + 1))
		bids = np.zeros((2, 2 * self.fi_obj.K + 1))
		p = [self.fi_obj.OB.prob_male, self.fi_obj.OB.prob_female]
		adval = self.fi_obj.ad_vals
		iters = 0
		while True:
			for gen in range(2):
				for diff in range(2 * self.fi_obj.K + 1):
					if (gen == 0 and diff == 2 * self.fi_obj.K) or (gen == 1 and diff == 0):
						u_new[gen, diff] = p[0] * u_old[0, diff] + p[1] * u_old[1, diff]
						bids[gen, diff] = 0.0
					else:
						win_diff = diff + (-1)**gen #new state
						term_2a = gam*(p[0]*u_old[0,win_diff] + p[1]*u_old[1,win_diff] - p[0]*u_old[0,diff] - p[1]*u_old[1,diff])
						term_2b = gam*(p[0]*u_old[0,diff] + p[1]*u_old[1,diff])
						(x, val) = get_discrete_maximizer(adval[gen]+term_2a, pwin_est[gen], self.fi_obj.OB.step_size)
						u_new[gen, diff] = val + term_2b
						bids[gen, diff] = x
			if bias_span(u_new.flatten() - u_old.flatten()) < epsilon_ev: #actually epsilon_ev should depend on t_k
				# print("span < epsilon_ev reached,  ", iters)
				break
			u_old = np.array(u_new)
			iters += 1
		return bids, u_new
	def PSRL_discrete(self, interval = 100, start = 0):
		def get_policy(Nbid_t_obs):
			N_adjusted = Nbid_t_obs + np.ones((2, self.fi_obj.OB.BB_n + 1))
			pwin_est = np.cumsum(N_adjusted, axis=1) / np.sum(N_adjusted, axis=1).reshape((2,1)) #g_\theta estimate
			ep_policy, _ = self.valiter(pwin_est)
			return ep_policy
		statetrack = start
		TU = 0.0
		t = 0
		Nbid_t_obs = np.zeros((2, self.fi_obj.OB.BB_n + 1)) #estimating the D_m and D_f, 
		per_step_reward = np.zeros(self.fi_obj.T) #reward in slot [0,T-1]
		for t in range(self.fi_obj.T):
			if t % interval == 0:  #recompute policy
				policy = get_policy(Nbid_t_obs)
			obs_gen = np.random.binomial(1, self.fi_obj.OB.prob_female)
			D = self.fi_obj.OB.draw_BBmax(obs_gen)
			bid_pos =  int (D / self.fi_obj.OB.step_size)
			Nbid_t_obs[obs_gen, bid_pos] += 1.0
			adval = self.fi_obj.ad_vals[obs_gen] 
			bid_t = policy[obs_gen, statetrack]
			if (statetrack == 2*self.fi_obj.K and obs_gen==0) or (statetrack == 0 and obs_gen == 1):
				bid_t = 0.0
			elif bid_t >= D:
				per_step_reward[t] += (adval - D)
				TU += per_step_reward[t]
				statetrack += (-1)**obs_gen
		self.TU_psrl = TU
		self.per_step_psrl = per_step_reward
	def km_psrl(self, interval = 100, start = 0):
		'''
			Using survival analysis techniques to estimate cdf
			with fixed interval policy recomputation
		'''
		statetrack = start
		#draws is of the form --> gender, bid bin, observationT1/observationT2
		draws = np.zeros((2, self.fi_obj.OB.BB_n + 1,2)) #can be optimized for others same value (case 2)
		e = np.array([[1, 0]] * (self.fi_obj.OB.BB_n + 1))
		TU = 0.0
		per_step_reward = np.zeros(self.fi_obj.T)
		t = 0
		while(True):
			if t % interval == 0:
				self.km_cdf_m = cdf_km_estimate(draws[0] + e, self.fi_obj.OB.step_size)
				self.km_cdf_f = cdf_km_estimate(draws[1] + e, self.fi_obj.OB.step_size)
				policy, _ = self.valiter(np.array([self.km_cdf_m, self.km_cdf_f]))
			obs_gen = np.random.binomial(1, self.fi_obj.OB.prob_female)
			adval = self.fi_obj.ad_vals[obs_gen]
			bid_t = policy[obs_gen, statetrack]
			D  = self.fi_obj.OB.draw_BBmax(obs_gen)
			if (statetrack == 2*self.fi_obj.K and obs_gen==0) or (statetrack == 0 and obs_gen == 1):
				bid_t = 0.0
				if bid_t == D:
					draws[obs_gen, 0, 0] += 1.0 #we view exact draw
			elif bid_t >= D:
				per_step_reward[t] += (adval - D)
				TU += per_step_reward[t] 
				statetrack += (-1)**obs_gen
				draws[obs_gen, int(D / self.fi_obj.OB.step_size), 0] += 1.0 #exact draw observed
			else: 
				index = int((bid_t+self.fi_obj.OB.step_size) / self.fi_obj.OB.step_size) 
				draws[obs_gen, index, 1] += 1.0 #censored draw observed
			t += 1
			# print("currentslot: ", t)
			if t >= self.fi_obj.T:
				break
		self.TU_kmpsrl = TU
		self.per_step_kmpsrl = per_step_reward