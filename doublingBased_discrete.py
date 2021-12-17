import numpy as np
from utility_functions import *
class DoublingBased:
	def __init__(self, fi_obj = None):
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
		
	def UCRL2_variant(self, start):
			def get_policy(Nbid_t_obs):
				N_adjusted = Nbid_t_obs + np.ones((2, self.fi_obj.OB.BB_n + 1))
				pwin_est = np.cumsum(N_adjusted, axis=1) / np.sum(N_adjusted, axis=1).reshape((2,1))
				ep_policy, _ = self.valiter(pwin_est)
				return ep_policy
			
			def not_doubled(v_k, N_tk):
				#v_k are counts in this episode
				return (v_k <= N_tk).all()
			
			statetrack = start
			TU = 0.0
			per_step_reward = np.zeros(self.fi_obj.T)
			t = 1
			ep = 1
			Nbid_t_obs = np.zeros((2, self.fi_obj.OB.BB_n + 1)) #actual draws saved in this
			while(True):
				v_k = np.zeros(2)
				N_tk = np.sum(Nbid_t_obs, axis=1) # num m, num f 
				policy_tk = get_policy(Nbid_t_obs) #policy for episode, i.e this inner while loop
				while t < self.fi_obj.T and not_doubled(v_k, N_tk):
					obs_gen = np.random.binomial(1, self.fi_obj.OB.prob_female)
					adval = self.fi_obj.ad_vals[obs_gen]
					v_k[obs_gen] += 1.0
					D_obs  = self.fi_obj.OB.draw_BBmax(obs_gen) #D_theta always observed
					bid_pos =  int (D_obs / self.fi_obj.OB.step_size)
					Nbid_t_obs[obs_gen, bid_pos] += 1.0 
					if (statetrack == 2*self.fi_obj.K and obs_gen==0) or (statetrack == 0 and obs_gen == 1):
						bid_t = 0.0
					else:
						bid_t = policy_tk[obs_gen, statetrack]
						if bid_t >= D_obs:
							per_step_reward[t] += (adval - D_obs)
							TU += per_step_reward[t]
							statetrack += (-1)**obs_gen
					t += 1
				# print("episode num: ", ep, " time slot: ", t)
				ep += 1
				if t >= self.fi_obj.T:
					break
			self.TU_UCRL2v = TU
			self.per_step_UCRL2v = per_step_reward
