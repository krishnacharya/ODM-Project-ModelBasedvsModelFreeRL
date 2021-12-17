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
perslot_reward_QL = np.zeros((n_sims, T))

for sim_ind in range(n_sims):
	learn = ModelFree(fi_obj = bf1)
	learn.Q_learning(start = K)
	perslot_reward_QL[sim_ind] = learn.per_step_QL
	print("Sim iteration #", sim_ind)
	print("Total reward Q_learning",learn.per_step_QL.sum(), learn.TU_QL)

# print("opt and simple discrete: -> ", bf1.val_tab_opt[-1,K], bf1.val_tab_det[-1,K])
with open("QL"+str(n_sims)+"sims"+case+".pickle","wb") as f:
	pickle.dump(perslot_reward_QL, f)

# with open("tmp.pickle", "rb") as f:
#     a,b = pickle.load(f) 