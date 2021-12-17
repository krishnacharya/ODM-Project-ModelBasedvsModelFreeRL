from OtherBidders import OtherBidders
from BidFairFullInfoDiscrete import BidFIDiscrete
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import *
import pickle
from config_forruns import *
 
ob1 = OtherBidders(prob_male = p_m, nOthers = nOthers)
ob1.set_cdf_BB(param_m = param[0], param_f = param[1])
bf1 = BidFIDiscrete(K=K, T = T, OB = ob1, ad_vals = ad_vals)

bf1.dp_bid_opt()
bf1.dp_bid_fixed(ad_vals) #truthful bidding.
print("Precomputed opt and simple discrete: -> ", bf1.val_tab_opt[-1,K], bf1.val_tab_det[-1,K],"\n") #val_tab[time steps rem,N_m-N_f+k,gender]

#put the above exact computation into a pickle file
# with open("FI-discreteexact"+case+".pickle", "wb") as f:
# 	pickle.dump((perslot_reward_opt, perslot_reward_simple), f)

#Validate precomputed policy's performance by running simulations:

bid_val = np.zeros((T, 2*K+1, 2))
bid_val[:,:,0] = ad_vals[0] # this is just bidding truthfully i.e own value.
bid_val[:,:,1] = ad_vals[1]
bid_val[:,2*K,0] = 0.0
bid_val[:,0,1] = 0.0

# n_sims = 50 #kept in config, alter if needed here
# perslot_reward_opt = np.zeros((n_sims, T)) #tracks reward in simulation #[0,nsims-1] for slot i \in [0,T-1]
# perslot_reward_simple = np.zeros((n_sims,T))
# for sim_ind in range(n_sims):
# 	state_track_opt = K #set to Nm - Nf = 0 at start of each simulation
# 	state_track_simple = K
# 	for i in range(T):
# 		obs_gen = np.random.binomial(1, 1-p_m) #gender drawn from bernoulli
# 		others_max = ob1.draw_BBmax(obs_gen)
# 		bid_opt = bf1.bid_tab_opt[T - i - 1, state_track_opt, obs_gen]
# 		bid_simple = bid_val[T - i - 1, state_track_simple, obs_gen]
# 		adval = ad_vals[obs_gen]
# 		if bid_opt >= others_max:
# 			perslot_reward_opt[sim_ind, i] += (adval - others_max)
# 			state_track_opt += (-1)**obs_gen
# 		if bid_simple >= others_max:
# 			perslot_reward_simple[sim_ind, i] += (adval - others_max)
# 			state_track_simple += (-1)**obs_gen
# 	print("Sim iteration #", sim_ind)
# 	print("Total reward opt",perslot_reward_opt[sim_ind].sum())
# 	print("Total reward simple", perslot_reward_simple[sim_ind].sum())

# # print("opt and simple discrete: -> ", bf1.val_tab_opt[-1,K], bf1.val_tab_det[-1,K])
# with open("FI-discrete"+str(n_sims)+"sims"+case+".pickle", "wb") as f:
# 	pickle.dump((perslot_reward_opt, perslot_reward_simple), f)

# with open("tmp.pickle", "rb") as f:
#     a,b = pickle.load(f) 