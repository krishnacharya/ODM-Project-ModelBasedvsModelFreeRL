from OtherBidders import OtherBidders
from BidFairFullInfoDiscrete import BidFIDiscrete
from ModelFree import ModelFree
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import *
import pickle
from config_forruns import *

ob1 = OtherBidders(prob_male = p_m, nOthers = nOthers)
ob1.set_cdf_BB(param_m = param[0], param_f = param[1])
bf1 = BidFIDiscrete(K=K, T = T, OB = ob1, ad_vals = ad_vals)

# n_sims = 50
perslot_reward_expsarsa = np.zeros((n_sims, T))

for sim_ind in range(n_sims):
	learn = ModelFree(fi_obj = bf1)
	learn.expected_sarsa(start = K)
	perslot_reward_expsarsa[sim_ind] = learn.per_step_Expsarsa
	print("Sim iteration #", sim_ind)
	print("Total reward ExpSarsa",learn.per_step_Expsarsa.sum(), learn.TU_ExpSarsa)

# print("opt and simple discrete: -> ", bf1.val_tab_opt[-1,K], bf1.val_tab_det[-1,K])
with open("ExpSarsa"+str(n_sims)+"sims"+case+".pickle","wb") as f:
	pickle.dump(perslot_reward_expsarsa, f)

# with open("tmp.pickle", "rb") as f:
#     a,b = pickle.load(f) 