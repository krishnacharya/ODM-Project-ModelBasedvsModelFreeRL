import numpy as np

def epsilon_greedy_choice(gen, diff, Q, epsilon):
	'''
	 wp epsilon explore any state
	 wp 1-epsilon chose max
	'''
	maxac_indices = np.argwhere(Q[gen, diff] == Q[gen,diff].max()).flatten()
	# other_indices = np.argwhere(Q[gen, diff] != Q[gen,diff].max()).flatten()
	flip = np.random.binomial(1, epsilon) # 1 wp epsilon
	if flip == 1: #explorew
		# if other_indices.size == 0:
		# 	return np.random.choice(maxac_indices) # if all are max, others is empty
		# return np.random.choice(other_indices)
		return np.random.randint(low=0, high=len(Q[gen, diff]))
	else:
		return np.random.choice(maxac_indices)
 
class ModelFree:
	'''
		Assuming all actions are discrete bids, 0.01,...1.00
	'''
	def __init__(self, fi_obj=None):
		'''
			fi_obj: Object of class BidFairFullInfo.
		'''
		self.fi_obj = fi_obj
	def Q_learning(self, epsilon = 0.1,alpha = 0.1, gam = 1.0, start = 0):
		'''
		Epsilon is used to choose the epsilon greedy strategy
		alpha is the step size
		gam is the discount factor - useful for convergence?
		Off policy TD, because we use max
		'''
		p = [self.fi_obj.OB.prob_male, self.fi_obj.OB.prob_female]
		c_diff = start #current difference n_m-n_f+K
		TU = 0.0
		#S - gender, diff, A - 0.00 0.01 ... 1.00
		Qtab = np.random.uniform(size = (2,2 * self.fi_obj.K + 1, self.fi_obj.OB.BB_n + 1)) # Q(State= (gender,diff),action = discrete bids)
		Qtab[0,2 * self.fi_obj.K,:] = 0.0
		Qtab[1,0,:] = 0.0
		t = 1
		per_step_reward = np.zeros(self.fi_obj.T)
		prev = np.random.binomial(1, self.fi_obj.OB.prob_female)
		while(t < self.fi_obj.T):
			cur_gen = prev
			next_gen = np.random.binomial(1, self.fi_obj.OB.prob_female) #next slot gender
			D_obs  = self.fi_obj.OB.draw_BBmax(cur_gen)
			adval = self.fi_obj.ad_vals[cur_gen]
			if (c_diff == 2*self.fi_obj.K and cur_gen==0) or (c_diff == 0 and cur_gen == 1):
				Qtab[cur_gen, c_diff,:] = (1-alpha)*Qtab[cur_gen, c_diff,:] + alpha*gam*np.max(Qtab[next_gen, c_diff])
				# Qtab[cur_gen, c_diff,:] = p[cur_gen] * Qtab[cur_gen,c_diff,:] + p[1-cur_gen] * np.max(Qtab[next_gen,c_diff])
				# Qtab[cur_gen,c_diff,:] = Qtab[cur_gen, c_diff,1] + \
				 							 # alpha*(gam*np.max(Qtab[next_gen, c_diff])- Qtab[cur_gen, c_diff,1]) #update all s,a similarly
			else:
				bid_index = epsilon_greedy_choice(cur_gen, c_diff, Qtab, epsilon)
				bid_t = bid_index * self.fi_obj.OB.step_size
				update = 0
				reward = 0
				if bid_t >= D_obs:
					reward = adval - D_obs
					per_step_reward[t] += reward
					TU += reward
					update = (-1)**cur_gen
				Qtab[cur_gen, c_diff, bid_index] = Qtab[cur_gen, c_diff, bid_index] + \
				alpha*(reward + gam*np.max(Qtab[next_gen,c_diff+update]) - Qtab[cur_gen,c_diff,bid_index])
				c_diff += update
			prev = next_gen		
			t += 1
		self.TU_QL = TU
		self.per_step_QL = per_step_reward

	def double_Q(self, epsilon = 0.1, alpha = 0.1, gam = 1.0, start = 0):
		def init_Q():
			Qtab = np.random.uniform(size = (2,2 * self.fi_obj.K + 1, self.fi_obj.OB.BB_n + 1))
			Qtab[0,2 * self.fi_obj.K,:] = 0.0
			Qtab[1,0,:] = 0.0
			return Qtab

		def Q_update(Qself, Qother, cur_gen, next_gen, c_diff, bid_index, update, reward):
			'''
				curr_gen, c_diff : S
				bidindex :A
				next_gen,c_diff+update: S'
			'''
			max_val = Qself[next_gen, c_diff+update].max()
			max_index = np.random.choice(np.argwhere(Qself[next_gen, c_diff+update] == max_val).flatten())
			Qself[cur_gen, c_diff, bid_index] = (1-alpha)*Qself[cur_gen, c_diff, bid_index] + \
				alpha*(reward + gam*Qother[next_gen, c_diff+update, max_index])

		p = [self.fi_obj.OB.prob_male, self.fi_obj.OB.prob_female]
		c_diff = start #current difference n_m-n_f+K
		TU = 0.0
		per_step_reward = np.zeros(self.fi_obj.T)
		Qtab1 = init_Q()
		Qtab2 = init_Q()
		t = 1
		prev = np.random.binomial(1, self.fi_obj.OB.prob_female)
		while(t < self.fi_obj.T):
			cur_gen = prev
			next_gen = np.random.binomial(1, self.fi_obj.OB.prob_female) #next slot gender
			Qup = np.random.binomial(1, 0.5) # if 0 update Qtab1, else update Qtab2
			D_obs  = self.fi_obj.OB.draw_BBmax(cur_gen)
			adval = self.fi_obj.ad_vals[cur_gen]
			if (c_diff == 2*self.fi_obj.K and cur_gen==0) or (c_diff == 0 and cur_gen == 1):
				if Qup == 0:
					max_index = np.random.choice(np.argwhere(Qtab1[next_gen, c_diff] == Qtab1[next_gen, c_diff].max()).flatten())
					Qtab1[cur_gen,c_diff,:] = (1-alpha)*Qtab1[cur_gen,c_diff,:] \
					+alpha*gam*Qtab2[next_gen, c_diff, max_index]
				else:
					max_index = np.random.choice(np.argwhere(Qtab2[next_gen, c_diff] == Qtab2[next_gen, c_diff].max()).flatten())
					Qtab2[cur_gen,c_diff,:] = (1-alpha)*Qtab2[cur_gen,c_diff,:] \
					+alpha*gam*Qtab1[next_gen, c_diff, max_index]
			else:
				bid_index = epsilon_greedy_choice(cur_gen, c_diff, Qtab1+Qtab2, epsilon)
				bid_t = bid_index * self.fi_obj.OB.step_size
				update = 0
				reward = 0
				if bid_t >= D_obs:
					reward = adval - D_obs
					per_step_reward[t] += reward
					TU += reward
					update = (-1)**cur_gen
				Q_update(Qtab1, Qtab2, cur_gen, next_gen, c_diff, bid_index, update, reward) if Qup == 0 else \
				Q_update(Qtab2, Qtab1, cur_gen, next_gen, c_diff, bid_index, update, reward)
				c_diff += update
			prev = next_gen		
			t += 1
		self.TU_doubleQ = TU
		self.per_step_DubQL = per_step_reward

	def sarsa(self, epsilon = 0.1,alpha = 0.1, gam = 1.0, start = 0):
		'''
			On policy TD
		'''
		c_diff = start
		TU = 0.0
		per_step_reward = np.zeros(self.fi_obj.T)
		Qtab = np.random.uniform(size = (2,2 * self.fi_obj.K + 1, self.fi_obj.OB.BB_n + 1))
		Qtab[0,2 * self.fi_obj.K,:] = 0.0
		Qtab[1,0,:] = 0.0
		prev = np.random.binomial(1, self.fi_obj.OB.prob_female)
		prev_bid_index = epsilon_greedy_choice(prev, c_diff, Qtab, epsilon) # A
		t = 1 
		while(t < self.fi_obj.T):
			cur_gen = prev # S = (cur_gen,c_diff)
			next_gen = np.random.binomial(1, self.fi_obj.OB.prob_female) # S' (g', diff'- yet to set)
			bid_index = prev_bid_index # A
			D_obs  = self.fi_obj.OB.draw_BBmax(cur_gen)
			adval = self.fi_obj.ad_vals[cur_gen]
			if (c_diff == 2*self.fi_obj.K and cur_gen==0) or (c_diff == 0 and cur_gen == 1):
				bid_index_next = epsilon_greedy_choice(next_gen, c_diff, Qtab, epsilon)
				Qtab[cur_gen, c_diff,:] = (1-alpha)*Qtab[cur_gen, c_diff,:] + \
										 alpha*gam*Qtab[next_gen, c_diff, bid_index_next]
				prev_bid_index = bid_index_next
				# Qtab[cur_gen,c_diff,:] = Qtab[cur_gen, c_diff,1] + alpha*(gam*np.max(Qtab[next_gen, c_diff])- Qtab[cur_gen, c_diff,1]) #update all s,a similarly
			else:
				bid_t = bid_index * self.fi_obj.OB.step_size
				update = 0
				reward = 0
				if bid_t >= D_obs:
					reward = adval - D_obs
					per_step_reward[t] += reward
					TU += reward
					update = (-1)**cur_gen # update to get S'
				bid_index_next = epsilon_greedy_choice(next_gen, c_diff+update, Qtab, epsilon) #get A' from S'
				Qtab[cur_gen, c_diff, bid_index] = Qtab[cur_gen, c_diff, bid_index] + \
				alpha*(reward + gam*Qtab[next_gen, c_diff+update, bid_index_next] - Qtab[cur_gen,c_diff,bid_index])
				c_diff += update
				prev_bid_index = bid_index_next
			prev = next_gen
			t += 1
		self.TU_sarsa = TU
		self.per_step_sarsa = per_step_reward

	def expected_sarsa(self, epsilon = 0.1,alpha = 1.0, gam = 1.0, start = 0):
		'''
			Structure almost like Q learning, but the for V_t use the expected
			value for e.g under epsilon greedy policy
		'''		
		def get_expectedVs_from_epG(gen, diff):
			'''
				V_s' = sum_a pi(s',a)Q(s',a)
				pi(s',a) is the epsilon greedy policy
				epsilon - random
				1-epsilon - pick max
			'''
			pi_next = np.full(Qtab[gen, diff].shape, epsilon / len(Qtab[gen, diff]))
			maxac_indices = np.argwhere(Qtab[gen, diff] == Qtab[gen,diff].max()).flatten()
			pi_next[maxac_indices] += (1.0 - epsilon) / len(maxac_indices)
			return np.dot(pi_next, Qtab[gen, diff])

		p = [self.fi_obj.OB.prob_male, self.fi_obj.OB.prob_female]
		c_diff = start #current difference n_m-n_f+K
		TU = 0.0
		per_step_reward = np.zeros(self.fi_obj.T)
		Qtab = np.random.uniform(size = (2,2 * self.fi_obj.K + 1, self.fi_obj.OB.BB_n + 1))
		Qtab[0,2 * self.fi_obj.K,:] = 0.0
		Qtab[1,0,:] = 0.0
		t = 1
		prev = np.random.binomial(1, self.fi_obj.OB.prob_female)
		while(t < self.fi_obj.T):
			cur_gen = prev
			next_gen = np.random.binomial(1, self.fi_obj.OB.prob_female) #next slot gender
			D_obs  = self.fi_obj.OB.draw_BBmax(cur_gen)
			adval = self.fi_obj.ad_vals[cur_gen]
			if (c_diff == 2*self.fi_obj.K and cur_gen==0) or (c_diff == 0 and cur_gen == 1):
				Qtab[cur_gen, c_diff,:] = (1-alpha)*Qtab[cur_gen, c_diff,:] + alpha*gam*get_expectedVs_from_epG(next_gen, c_diff)
			else:
				bid_index = epsilon_greedy_choice(cur_gen, c_diff, Qtab, epsilon)
				bid_t = bid_index * self.fi_obj.OB.step_size
				update = 0
				reward = 0
				if bid_t >= D_obs:
					reward = adval - D_obs
					per_step_reward[t] += reward
					TU += reward
					update = (-1)**cur_gen
				Qtab[cur_gen, c_diff, bid_index] = Qtab[cur_gen, c_diff, bid_index] + \
				alpha*(reward + gam*get_expectedVs_from_epG(next_gen, c_diff+update) - Qtab[cur_gen,c_diff,bid_index])
				c_diff += update
			prev = next_gen
			t += 1
		self.TU_ExpSarsa = TU
		self.per_step_Expsarsa = per_step_reward