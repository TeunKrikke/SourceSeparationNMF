import theano as th
from theano import tensor as T

from utils import normal, convolution, normalise_W, normalise_H, shift

def update_w_beta(V, W, H, beta=0, conv=False, norm_W=0, K=0, norm_size=0):
    """
     Update W according the beta divergence

     Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
     norm_W -- normalise the W matrix with L1 norm (1) or L2 norm (2) (default 0)
     beta -- the beta parameter that determines which update rule to use Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
    """

    if conv:
        V_hat = convolution(W,H,K)
         
        one_vshape =  T.ones(V.shape)
        for k in range(K):
            if beta == 2:
            # W = W .* (V*H')./(W*(H*H'))
                W = T.mul(W, T.dot(V, T.transpose(H))/T.dot(W,T.dot(H, T.transpose(H))))

            # beta = 1
            # W = W .* ((V./V_ap)*H.')./repmat(sum(H,2)',F,1);
            # scale = sum(W,1);
            elif beta == 1:
                # W = T.set_subtensor(W, T.mul(W, T.dot((V/V_hat), T.transpose(shift(H, k))) / T.extra_ops.repeat(shift(H, k).sum(axis=1), V.shape[0]).reshape(W.shape)))
                # W[:,:,k] = T.mul(W, T.dot((V/V_hat), T.transpose(shift(H, k))) / T.extra_ops.repeat(shift(H, k).sum(axis=1), V.shape[0]).reshape(W.shape))
                W = T.set_subtensor(W[:,:,k], T.mul(W[:,:,k], T.dot((V/V_hat), T.transpose(shift(H, k)))/ T.dot(one_vshape,T.transpose(shift(H,k)))))
            # beta = 0
            else:
                W = T.set_subtensor(W[:,:,k], T.mul(W[:,:,k], T.dot(T.mul(V, T.power(V_hat,beta-2)), T.transpose(shift(H, k))) / T.dot(T.power(V_hat,beta-1), T.transpose(shift(H, k)))))
    else:
        V_hat = normal(W,H)
        # beta = 2
        if beta == 2:
        # W = W .* (V*H')./(W*(H*H'))
            W = T.mul(W, T.dot(V, T.transpose(H))/T.dot(W,T.dot(H, T.transpose(H))))

        # beta = 1
        # W = W .* ((V./V_ap)*H.')./repmat(sum(H,2)',F,1);
        # scale = sum(W,1);
        elif beta == 1:
            W = T.mul(W, T.dot((V/V_hat), T.transpose(H)) / T.extra_ops.repeat(H.sum(axis=1), V.shape[0]).reshape(W.shape))

        # beta = 0
        else:
            W = T.mul(W, T.dot(T.mul(V, T.power(V_hat,beta-2)), T.transpose(H)) / T.dot(T.power(V_hat,beta-1), T.transpose(H)))
        
    if norm_W > 0:
        return normalise_W(W, norm_W, norm_size)
    else:
        return W
    
def update_h_beta(V, W, H, beta=0, conv=False, norm_H=0, K=0, norm_size=0):
    """
     Update H according the beta divergence

     Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
     norm_H -- normalise the H matrix with L1 norm (1) or L2 norm (2) (default 0)
     beta -- the beta parameter that determines which update rule to use Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
    """
    
    if conv:
        
        one_vshape =  T.ones(V.shape)
        V_hat = convolution(W,H, K)

        H_old = H
        H = T.zeros(H.shape)

        for k in range(K):
            # beta = 2
            if beta == 2:
                # H = H .* (W'* V)./((W'*W)*H);
                H = T.mul(H, T.dot(T.transpose(W), V)/T.dot(T.dot(T.transpose(W), W), H))

            # beta = 1
            elif beta == 1:
            # H = H .* (W.'*(V./V_ap))./repmat(scale',1,N);
                H = H + T.mul(H_old, (T.dot(T.transpose(W[:,:,k]), shift((V/V_hat), -k)) / T.dot(T.transpose(W[:,:,k]), one_vshape)))
                
            #beta = 0
            else:
                H = H + T.mul(H_old, T.dot(T.transpose(W[:,:,k]), T.mul(V, T.power(V_hat,beta-2))) / T.dot(T.transpose(W[:,:,k]), T.power(V_hat,beta-1)))
    else:
        V_hat = normal(W,H)
        # beta = 2
        if beta == 2:
            # H = H .* (W'* V)./((W'*W)*H);
            H = T.mul(H, T.dot(T.transpose(W), V)/T.dot(T.dot(T.transpose(W), W), H))

        # beta = 1
        elif beta == 1:
        # H = H .* (W.'*(V./V_ap))./repmat(scale',1,N);
            scale = W.sum(axis=0)
            H = T.mul(H, T.dot(T.transpose(W), (V/V_hat)) / T.extra_ops.repeat(T.transpose(scale), V.shape[1]).reshape(H.shape))
        #beta = 0
        else:
            H = T.mul(H, T.dot(T.transpose(W), T.mul(V, T.power(V_hat,beta-2))) / T.dot(T.transpose(W), T.power(V_hat,beta-1)))
        
    if norm_H > 0:
        return normalise_H(H, norm_H, norm_size)
    else:
        return H

def update_w_cauchy(V, W, H, beta=0, conv=False, norm_W=0, K=0, norm_size=0):
    """
     Update W according for cauchy

     Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
     norm_W -- normalise the W matrix with L1 norm (1) or L2 norm (2) (default 0)
     beta -- the beta parameter that determines which update rule to use Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
    """
    if conv:
        V_hat = convolution(W,H, K)
        z = 3 * V_hat / (T.power(V, 2) + T.power(V_hat, 2))

        for k in range(K):
            # W = T.mul(W, T.dot(T.power(V_hat,-1), T.transpose(H)) / T.dot(z, T.transpose(H)))
            W = T.set_subtensor(W[:,:,k], T.mul(W[:,:,k], T.dot(T.power(V_hat,-1), T.transpose(H)) / T.dot(z, T.transpose(H))))
    else:
        V_hat = normal(W,H)
    
        z = 3 * V_hat / (T.power(V, 2) + T.power(V_hat, 2))
        W = T.mul(W, T.dot(T.power(V_hat,-1), T.transpose(H)) / T.dot(z, T.transpose(H)))
    
    if norm_W > 0:
        return normalise_W(W, norm_W, norm_size)
    else:
        return W

def update_h_cauchy(V, W, H, beta=0, conv=False, norm_H=0, K=0, norm_size=0):
    """
     Update H according for cauchy

     Keyword arguments:
     V -- the magnitude of the input signal
     W -- the frequency matrix for each source F x K
     H -- the activation matrix for each source K x N
     convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
     norm_H -- normalise the H matrix with L1 norm (1) or L2 norm (2) (default 0)
     beta -- the beta parameter that determines which update rule to use Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
    """
    if conv:
        V_hat = convolution(W,H,K)

        z = 3 * V_hat / (T.power(V, 2) + T.power(V_hat, 2))
        H_old = H
        H = T.zeros(H.shape)

        for k in range(K):
            H = T.mul(H, T.dot(T.transpose(W[:,:,k]), T.power(V_hat,-1)) / T.dot(T.transpose(W[:,:,k]), z))
    else:
        V_hat = normal(W,H)
    
        z = 3 * V_hat / (T.power(V, 2) + T.power(V_hat, 2))
        
        H = T.mul(H, T.dot(T.transpose(W), T.power(V_hat,-1)) / T.dot(T.transpose(W), z))
    
    if norm_H > 0:
        return normalise_H(H, norm_H, norm_size)
    else:
        return H
