import theano as th
from theano import tensor as T
from utils import normal, convolution

def cost_is(V, W, H, frequencies, time_steps, conv=False):
    """
     Itakura-Saito cost function
    
    Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     frequencies -- the F side of the V and W matrix needed to scale the answer of the IS function
     time_steps -- the N side of the V and H matrix needed to scale the answer of the IS function
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
     
    """
    if conv:
        V_hat = convolution(W,H)
    else:
        V_hat = normal(W,H)
    return T.sum(V/V_hat - T.log(V/V_hat)) - frequencies/time_steps

       
def cost_cau(V, W, H, frequencies, time_steps, conv=False):
    """
     Cauchy cost function
    
    Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     frequencies -- the F side of the V and W matrix not needed here
     time_steps -- the N side of the V and H matrix not needed here
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
    """
    if conv:
        V_hat = convolution(W,H)
    else:
        V_hat = normal(W,H)
    return T.sum(3/2 * T.log(T.power(V, 2) + T.power(V_hat, 2)) - T.log(V_hat))

def cost_euc(V, W, H, frequencies, time_steps, conv=False):
    """
     Euclidean cost function
    
    Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     frequencies -- the F side of the V and W matrix not needed here
     time_steps -- the N side of the V and H matrix not needed here
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
    """
    if conv:
        V_hat = convolution(W,H)
    else:
        V_hat = normal(W,H)

    return T.sum(0.5 * T.sqr(V - V_hat))

def cost_kl(V, W, H, frequencies, time_steps, conv=False):
    """
     Kullback-Leibler cost function
    
    Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     frequencies -- the F side of the V and W matrix not needed here
     time_steps -- the N side of the V and H matrix not needed here
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
    """
    if conv:
        V_hat = convolution(W,H)
    else:
        V_hat = normal(W,H)

    return T.sum(V * T.log(V/V_hat) - (V + V_hat))