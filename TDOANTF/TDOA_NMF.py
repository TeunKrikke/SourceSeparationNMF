import numpy as np
import scipy.linalg as linalg
from sklearn.cluster import KMeans

import time

# from utils import stft, istft, write_wav, read_wav
from utils_reverb import load_files, do_reverb, do_stft, signum
import matplotlib.pyplot as plt

import mir_eval

class TDOA_NMF(object):
    def __init__(self, X, mic_locs, WINDOW_SIZE=1024,
                 max_az=5, max_th=5, K=2, fs=16000):
        self.M, self.F, self.N = X.shape # mics
        self.MM = self.M * self.M
        self.max_theta = max_th
        self.max_azimuth = max_az
        self.radius = 1
        self.O = max_az * max_th # look directions ie azimuth * theta
        self.K = K
        self.X = np.zeros((self.F, self.N, self.MM), dtype="complex")
        self.X[:,:,0] = X[0,:,:] * np.conj(X[0,:,:])
        self.X[:,:,1] = X[0,:,:] * np.conj(X[1,:,:])
        self.X[:,:,2] = X[1,:,:] * np.conj(X[0,:,:])
        self.X[:,:,3] = X[1,:,:] * np.conj(X[1,:,:])



        self.determine_A(self.F, fs, WINDOW_SIZE, mic_locs)
        self.norm_A()

        self.Q = np.random.random((self.K, self.O))
        self.W = np.random.random((self.F, self.K))
        self.H = np.random.random((self.K, self.N))

    def determine_A(self, F, fs, N, MIC_LOCS):
        speed_of_sound = 344
        self.A = np.zeros((self.F, self.O, np.power(len(MIC_LOCS),2)), dtype="complex")

        azimuth = 0
        theta = 0
        for ta in range(self.max_theta * self.max_azimuth):
            if azimuth == self.max_azimuth:
                theta += 1
                azimuth = 0
            n = 0
            m = 0
            k_o = np.asarray([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(azimuth)), self.radius])
            for nm in range(len(MIC_LOCS) * len(MIC_LOCS)):
                if m == len(MIC_LOCS):
                    n += 1
                    m = 0
                tav_n = np.negative(k_o).T.dot(MIC_LOCS[n]) / speed_of_sound
                tav_m = np.negative(k_o).T.dot(MIC_LOCS[m]) / speed_of_sound
                for i in range(F):
                    f_i = (i - 1) * fs / N
                    self.A[i, ta, nm] = np.exp(1j * 2 * np.pi * f_i * tav_n - tav_m)
            azimuth += 1

    def calc_xhat_E(self):
        x_hat = np.zeros((self.F, self.N), dtype=self.W.dtype)
        E = np.zeros((self.F, self.N, self.MM), dtype=self.A.dtype)
        WH = np.matmul(self.W, self.H)

        for o in range(self.O):
            A_o = self.A[:,o,:].reshape(self.F, self.MM, 1)
            for k in range(self.K):

                x_hat_ko = self.Q[k, o] * WH #FN1
                x_hat += x_hat_ko

                E += np.matmul(A_o, x_hat_ko.reshape(self.F,1,self.N)).reshape(self.F,self.N,self.MM)

        E = self.X - E
        tr = np.sum(np.trace(np.matmul(E, self.A.reshape(self.F,self.MM,self.O))))

        return x_hat, E, tr.real

    def norm_A(self):
        F, O, M = self.A.shape
        for mic in range(self.M):
            A_nm = self.A[:,:,mic].reshape(self.F, self.O)
            self.A[:,:,mic] = np.divide(A_nm, linalg.norm(A_nm))

    def run(self, epochs=100):
        for epoch in range(epochs):
            start1 = time.time()
            x_hat, E, tr = self.calc_xhat_E()

            QH = np.sum(self.Q,axis=1) * np.sum(self.H,axis=1)
            self.W = self.W * (1 + ((QH * tr) / (QH * np.sum(x_hat, axis=1).reshape(self.F,1))))

            x_hat, E, tr = self.calc_xhat_E()

            QW = (np.sum(self.Q, axis=1) * np.sum(self.W, axis=0) * tr).reshape(1,self.K)
            self.H = self.H * (1 + ((QW * tr)/(QW * np.sum(x_hat, axis=0).reshape(self.N,1))).T)

            a_hat = np.sum(np.power(self.H,2), axis=1).reshape(self.K,1)
            self.H = self.H / a_hat
            self.W = self.W * a_hat.T

            x_hat, E, tr = self.calc_xhat_E()

            WH = np.matmul(self.W, self.H)
            self.Q = self.Q * (1+np.sum((WH * tr) / (WH * x_hat)))
            b_hat = np.sqrt(np.sum(self.Q**2,axis=1)).reshape(self.K,1)
            self.Q = self.Q / b_hat
            self.W = self.W * b_hat.T

            x_hat, E, tr = self.calc_xhat_E()
            WH = np.matmul(self.W, self.H)
            WHV = np.sum(np.multiply(WH, x_hat), axis=1).reshape(1,self.F)
            WHE = np.sum(np.multiply(WH.reshape(self.F,self.N,1), E), axis=1).reshape(1,self.F,self.M*self.M)
            Q_k = np.sum(self.Q,axis=0).reshape(self.O,1)

            A_hat = np.multiply(self.A,np.einsum('kn,nkm->nkm',
                                            np.matmul(Q_k, WHV),
                                            np.einsum('kj,jnm->nkm',Q_k, WHE)))

            for m in range(self.M):
                LV, D, RV = linalg.svd(A_hat[:,:,m])
                D_hat = np.zeros((self.F, self.O), dtype=A_hat.dtype)
                D_hat[:self.O,:self.O] = np.diag(D)
                D_hat[D_hat < 0] = 0
                A_hat_m = np.matmul(LV, np.matmul(D_hat, RV))

                self.A[:,:,m] = np.multiply(np.absolute(A_hat[:,:,m]),
                                       np.exp(1j * np.angle(self.A[:,:,m])))
            self.norm_A()

            x_hat, E, tr = self.calc_xhat_E()
            print(np.linalg.norm(E))


    def reconstruct(self, x, sources=2):
        small_Q = self.Q[0,:].reshape(-1,1)
        b = np.zeros((sources,self.K))

        single_b = KMeans(n_clusters=sources, random_state=0).fit_predict(np.real(self.Q))
        for k in range(self.K):
            b[single_b[k], k] = 1

        WH = np.matmul(self.W, self.H)
        BQ = np.matmul(b,self.Q)
        S = np.zeros((self.F, self.N, sources), dtype=self.A.dtype)
        S_full = np.zeros((self.F, self.N), dtype=self.A.dtype)
        for q in range(sources):
            for o in range(self.O):
                S[:, :, q] += BQ[q, o] * WH
                S_full[:, :] += S[:, :, q]

        Y = np.zeros((self.F, self.N, sources), dtype=self.A.dtype)
        for q in range(sources):
            Y[:,:,q] = x * (S[:,:,q] / S_full)

        return Y

if __name__ == '__main__':
    files = ['/home/tinus/Workspace/corpus/TIMIT/TRAIN/DR1/FKFB0/SA1.WAV', '/home/tinus/Workspace/corpus/TIMIT/TRAIN/DR1/FDML0/SA1.WAV']
    s1, s2 = load_files(files)
    room, locs = do_reverb(s1, s2)
    Y1, Y2, X1, X2 = do_stft(s1,s2,room)
    X = np.asarray([signum(Y1),signum(Y2)])
    nmf = TDOA_NMF(X, locs, K=3)
    nmf.run(epochs=20)
    nmf.reconstruct(Y1, sources=2)
