from OtherBidders import OtherBidders
from BidFairFullInfoDiscrete import BidFIDiscrete
from PSRLtype_discrete import PSRLTypeAlgos
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import *
import pickle
from config_forruns import *

ob1 = OtherBidders(prob_male = p_m, nOthers = nOthers)
ob1.set_cdf_BB(param_m = param[0], param_f = param[1])
bf1 = BidFIDiscrete(K=K, T = T, OB = ob1, ad_vals = ad_vals)

# n_sims = 20
perslot_reward_psrl = np.zeros((n_sims, T))

for sim_ind in range(n_sims):
	learn = PSRLTypeAlgos(fi_obj = bf1)
	learn.PSRL_discrete(start=K)
	perslot_reward_psrl[sim_ind] = learn.per_step_psrl
	print("Sim iteration #", sim_ind)
	print("Total reward PSRL",learn.per_step_psrl.sum(), learn.TU_psrl)

# print("opt and simple discrete: -> ", bf1.val_tab_opt[-1,K], bf1.val_tab_det[-1,K])
with open("PSRL"+str(n_sims)+"sims"+case+".pickle","wb") as f:
	pickle.dump(perslot_reward_psrl, f)

# with open("tmp.pickle", "rb") as f:
#     a,b = pickle.load(f) 