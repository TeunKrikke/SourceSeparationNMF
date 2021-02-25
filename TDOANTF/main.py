import numpy as np
import scipy.linalg as linalg
from sklearn.cluster import KMeans

import time

from utils import stft, istft, write_wav, read_wav, determine_A
import matplotlib.pyplot as plt

import mir_eval

def calc_xhat_E(Q, W, H, X, A):
    K, O = Q.shape
    F, N, M = X.shape
    #A = FOM
    x_hat = np.zeros((F,N), dtype=A.dtype)
    E = np.zeros((F, N, M), dtype=A.dtype)
    WH = np.matmul(W, H)
    start = time.time()
    for o in range(O):
        A_o = A[:,o,:].reshape(F, M, 1)
        for k in range(K):

            x_hat_ko = Q[k, o] * WH #FN1
            x_hat += x_hat_ko

            E += np.matmul(A_o, x_hat_ko.reshape(F,1,N)).reshape(F,N,M)

    print("calc_xhat_E: "+str(time.time()-start))

    E = X - E
    tr = np.sum(np.trace(np.matmul(E, A.reshape(F,M,O))))

    return x_hat, E, tr

def norm_A(A):
    F, O, M = A.shape
    for mic in range(M):
        A_nm = A[:,:,mic].reshape(F, O)
        A[:,:,mic] = np.divide(A_nm, linalg.norm(A_nm))
    return A

def main():
    start = time.time()
    max_az = 5
    max_th = 5
    M = 3 # mics
    O = max_az * max_th # look directions ie azimuth * theta
    K = 2
    F = 513 # frequency bins
    N = 5462 # time steps
    WINDOW_SIZE = 1024
    fs = 16000
    iterations = 200
    epochs = 2

    X = np.random.random((F,N,M*M))

    A_ION = determine_A(F, fs, WINDOW_SIZE, max_azimuth=max_az, max_theta=max_th)
    A = norm_A(A_ION)

    Q = np.random.random((K, O))
    W = np.random.random((F, K))
    H = np.random.random((K, N))
    print("starting loop")
    #start loop
    for epoch in range(epochs):
        start1 = time.time()
        x_hat, E, tr = calc_xhat_E(Q, W, H, X, A)


        QH = np.sum(Q,axis=1) * np.sum(H,axis=1)
        W = W * (1 + ((QH * tr) / (QH * np.sum(x_hat, axis=1).reshape(F,1))))
        print(time.time()-start1)

        x_hat, E, tr = calc_xhat_E(Q, W, H, X, A)

        QW = (np.sum(Q, axis=1) * np.sum(W, axis=0) * tr).reshape(1,K)
        H = H * (1 + ((QW * tr)/(QW * np.sum(x_hat, axis=0).reshape(N,1))).T)
        print(time.time()-start1)
        a_hat = np.sum(np.power(H,2), axis=1).reshape(K,1)
        H = H / a_hat
        W = W * a_hat.T

        x_hat, E, tr = calc_xhat_E(Q, W, H, X, A)

        Q = Q * (1+np.sum((np.matmul(W, H) * tr) / (np.matmul(W,H) * x_hat)))
        print(time.time()-start1)
        b_hat = np.sqrt(np.sum(Q**2,axis=1)).reshape(K,1)
        Q = Q / b_hat
        W = W * b_hat.T

        x_hat, E, tr = calc_xhat_E(Q, W, H, X, A)
        print(time.time()-start1)
        WH = np.matmul(W, H)
        WHV = np.sum(np.multiply(WH, x_hat), axis=1).reshape(1,F)
        WHE = np.sum(np.multiply(WH.reshape(F,N,1), E), axis=1).reshape(1,F,M*M)
        Q_k = np.sum(Q,axis=0).reshape(O,1)

        A_hat = np.multiply(A,np.einsum('kn,nkm->nkm',
                                        np.matmul(Q_k, WHV),
                                        np.einsum('kj,jnm->nkm',Q_k, WHE)))

        for m in range(M):
            LV, D, RV = linalg.svd(A_hat[:,:,m])
            D_hat = np.zeros((F, O), dtype=A_hat.dtype)
            D_hat[:O,:O] = np.diag(D)
            D_hat[D_hat < 0] = 0
            A_hat_m = np.matmul(LV, np.matmul(D_hat, RV))

            A[:,:,m] = np.multiply(np.absolute(A_hat[:,:,m]),
                                   np.exp(1j * np.angle(A[:,:,m])))
        A = norm_A(A)


    print(time.time()-start)


if __name__ == '__main__':
    main()
