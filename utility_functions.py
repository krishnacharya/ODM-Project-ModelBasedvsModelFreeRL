import math
import numpy as np
def round_nearest(x, a):
    return round(x / a) * a

def round_down(x, a):
    return math.floor(x / a) * a

def round_up(x, a):
    return math.ceil(x / a) * a

def step_wise_integral(x, f, step_size):
    '''
        To avoid scipys integrate.quad warning messages
        This computes integral of f(a cdf) from 0 to x

        f: a stepwise cdf
        step_size: is the size of steps of this stepwise funtion.

    '''
    integral_val = 0.0
    if x >= 1.0:
        for i in range(int(1.0 / step_size)):
            integral_val += f[i] * step_size
        integral_val += (x - 1.0)
    elif x <= 0:
        return 0.0
    else:
        index = int (round_down(x, step_size) / step_size)
        for i in range(index):
            integral_val += f[i] * step_size
        integral_val += f[index] * (x - step_size*index) #see for e.g x = 0.36, round down to 0.3
    return integral_val
def get_pmf(cdf):
	'''
		get pmf from cdf discrete
		cdf: is a numpy array
	'''
	l = len(cdf)
	return np.append(cdf[0], cdf[1:l] - cdf[:l-1])
    
def get_cdf(pmf):
	return np.cumsum(pmf, axis=1)#pmf is [pmfmale,pmffemale], to change


def bias_span(vec):
    return np.max(vec) - np.min(vec)

def cdf_km_estimate(draws, step_size):
    """
        draws looks like
            0.0| 0.01 | 0.02 ....
       ED   10   4
       CD   1    2
    """   
    ccdf = np.zeros(int(1.0 / step_size) + 1) #complement of estimated cdf.
    dge = np.sum(draws) #draws >=
    prod = 1.0
    for i in range(len(ccdf)):
        d_i = draws[i][0]
        if d_i > 0:
            y_i = dge
            if dge != np.sum(draws[i:]):
                print("Error!")
            prod *= (1 - d_i/y_i)
            ccdf[i] = prod
        else:
            ccdf[i] = prod
        dge -= draws[i,0] + draws[i,1]
    return 1-ccdf

def not_doubled(v_k, N_tk):
    #v_k are counts in this episode
    return (v_k <= N_tk).all()

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