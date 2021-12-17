import numpy as np
essential_alphabeta = np.array([[2,47],[4,34],[9,38],[15,38],[22,36],[16,19],[27,20],[25,12],[27,7]])
K = 5

nOthers = 49 #number of bidders other than our constrained bidder j
# T = 10**4
T = 10**4
p_m = 0.5 #probability of male user appearing = 0.5, probability of female user  = 1-pm
# p_m = 0.1
# param = [essential_alphabeta[1], essential_alphabeta[2]] #others 0.2, 0.3
# case = "P1-100bidders"
# param = [essential_alphabeta[5], essential_alphabeta[7]] #others 0.6,0.8
# case = "P2"
# param = [essential_alphabeta[4],essential_alphabeta[5]] #others 0.5,0.6
# case = "P3"
# param = [essential_alphabeta[0], essential_alphabeta[7]] #example1 - 0.1,0.8
# param = [essential_alphabeta[2], essential_alphabeta[7]] #others 0.3,0.8
# case = "P4-v2good"
# case = "P4new-50bidders"

# param = [essential_alphabeta[1], essential_alphabeta[7]] #others 0.2,0.2 


#Pos cases
param = [essential_alphabeta[0],essential_alphabeta[5]] # 0.1, 0.6 (overbid for female to maintain parity?, PosCase1, opt much better)
# param = [essential_alphabeta[2],essential_alphabeta[3]] # 0.3,0.4 (truthful bidding also performs well)
# param = [essential_alphabeta[5], essential_alphabeta[7]] #0.6, 0.8 (truthful and optimal both poor)

case = "SameValtest"
# ad_vals = np.array([0.4, 0.7])
ad_vals = np.array([0.5, 0.5]) #V_m, V_f = 0.5, i.e. advertiser j values male and female equally
n_sims = 50