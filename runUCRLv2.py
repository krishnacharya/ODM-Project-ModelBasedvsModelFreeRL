from OtherBidders import OtherBidders
from BidFairFullInfoDiscrete import BidFIDiscrete
from doublingBased_discrete import DoublingBased
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import *
import pickle
from config_forruns import *

ob1 = OtherBidders(prob_male = p_m, nOthers = nOthers)
ob1.set_cdf_BB(param_m = param[0], param_f = param[1])
bf1 = BidFIDiscrete(K=K, T = T, OB = ob1, ad_vals = ad_vals)

# n_sims = 50
perslot_reward_UCRLv2 = np.zeros((n_sims, T))

for sim_ind in range(n_sims):
	learn = DoublingBased(fi_obj = bf1)
	learn.UCRL2_variant(start=K)
	perslot_reward_UCRLv2[sim_ind] = learn.per_step_UCRL2v
	print("Sim iteration #", sim_ind)
	print("Total reward UCRLv2",learn.per_step_UCRL2v.sum(), learn.TU_UCRL2v)

# print("opt and simple discrete: -> ", bf1.val_tab_opt[-1,K], bf1.val_tab_det[-1,K])
with open("UCRLv2"+str(n_sims)+"sims"+case+".pickle","wb") as f:
	pickle.dump(perslot_reward_UCRLv2, f)

# with open("tmp.pickle", "rb") as f:
#     a,b = pickle.load(f) 