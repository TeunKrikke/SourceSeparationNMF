import numpy as np
import scipy.linalg as linalg
from sklearn.cluster import KMeans

import time

import corpus

from librosa import istft

# from utils import stft, istft, write_wav, read_wav
from utils_reverb import load_files, do_reverb, do_stft, do_stft_2mic
import matplotlib.pyplot as plt

import mir_eval

def signum(x):
    return x/np.abs(x)

class SCM_CNTF(object):
    def __init__(self, X, mic_locs, K=2,
                 fs=16000, win_size=1024, max_az=5, max_th=5, radius=1,
                 DEBUG=True):
        self.X = X
        self.F, self.T, self.M, self.M = self.X.shape
        self.O = max_az * max_th
        self.radius = radius
        self.K = K
        self.DEBUG = DEBUG

        self.Q = np.random.random((self.K, self.O))
        self.W = np.random.random((self.F, self.K))
        self.H = np.random.random((self.K, self.T))


        self.determineA(fs, win_size, max_az, max_th, mic_locs)
        self.normaliseA()


    def determineA(self, fs, N, max_azimuth, max_theta, mic_locs):
        v = 344
        self.A = np.zeros((self.F, self.O, self.M, self.M), 'complex')
        azimuth = 0
        theta = 0

        for o in range(self.O):
            if azimuth == max_azimuth:
                theta += 1
                azimuth = 0
            k_o = np.asarray([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(azimuth)), self.radius])
            for f in range(self.F):
                f_i = f * fs / N

                for n in range(self.M):
                    for m in range(self.M):
                        tav_n = np.matmul(np.negative(k_o).T, mic_locs[n])/v
                        tav_m = np.matmul(np.negative(k_o).T, mic_locs[m])/v
                        tav = tav_n - tav_m
                        self.A[f,o,n,m] = np.exp(1j * 2 * np.pi * f_i * tav)
            azimuth += 1

    def normaliseA(self):
        for f in range(self.F):
            for o in range(self.O):
                self.A[f, o, :, :] = np.divide(self.A[f, o, :, :],
                                               np.linalg.norm(self.A[f, o, :, :]))

    def calc_E(self):
        start = time.time()
        AoQ = np.zeros([self.F, self.O, self.M, self.M, self.K], self.X.dtype)
        xhat = np.zeros([self.O, self.F, self.K], self.X.dtype)

        for o in range(self.O):
            for k in range(self.K):
                AoQ[:, o, :, :, k] = np.multiply(self.A[:,o,:,:], self.Q[k,o])
                xhat[o,:,k] = np.multiply(self.Q[k,o], self.W[:,k])

        for f in range(self.F):
            for k in range(self.K):
                AoQ[f,:,:,:,k] = np.multiply(AoQ[f,:,:,:,k], self.W[f,k])

        Xhat = np.matmul(AoQ, self.H)
        xhat = np.matmul(np.sum(xhat,axis=0), self.H)

        Xhat = np.sum(np.swapaxes(Xhat, -1, 1), axis=-1)

        E = self.X - Xhat

        # print("calc_E  took " + str(time.time()-start))

        return E, xhat

    def updateW(self):
        E, xhat = self.calc_E()
        EoA = np.swapaxes(self.calc_trace_EoA(E), 0, -1)
        QoH = np.zeros((self.K,self.O, self.T), 'complex')
        upper = np.zeros((self.O, self.F, self.K), 'complex')
        lower = np.zeros((self.O, self.F, self.K), 'complex')
        for o in range(self.O):
            for k in range(self.K):
                QoH[k,o,:] = np.multiply(self.Q[k,o], self.H[k,:])

        for o in range(self.O):
            QoH_o = QoH[:,o,:]
            EoA_o = EoA[o,:,:]
            upper[o,:,:] = np.swapaxes(np.matmul(QoH_o,
                                                   EoA_o),
                                         0,1)
            lower[o,:,:] = np.swapaxes(np.matmul(QoH_o,
                                                   np.swapaxes(xhat,
                                                               0,1)),
                                         0,1)
        upper = np.sum(upper,axis=0)
        lower = np.sum(lower,axis=0)

        self.W = np.multiply(self.W, 1 + np.divide(upper, lower))



    def updateH(self):
        E, xhat = self.calc_E()
        EoA = self.calc_trace_EoA(E)
        QoW = np.zeros((self.K, self.O, self.F), 'complex')
        upper = np.zeros((self.O, self.K, self.T), 'complex')
        lower = np.zeros((self.O, self.K, self.T), 'complex')
        for o in range(self.O):
            for k in range(self.K):
                QoW[k,o,:] = np.multiply(self.Q[k,o], self.W[:,k])

        for o in range(self.O):
            QoW_o = QoW[:,o,:]
            EoA_o = EoA[:,:,o]
            upper[o,:,:] = np.matmul(QoW_o, EoA_o)

            lower[o,:,:] = np.matmul(QoW_o,xhat)

        upper = np.sum(upper,axis=0)
        lower = np.sum(lower,axis=0)

        self.H = np.multiply(self.H, 1 + np.divide(upper, lower))

    def calc_trace_EoA(self, E):
        EoA = np.zeros((self.F, self.T, self.O), E.dtype)

        for f in range(self.F):
            for t in range(self.T):
                EoA[f,t,:] = np.trace(np.multiply(E[f,t], self.A[f,:]), axis1=1, axis2=2)

        return EoA

    def updateQ(self):
        E, xhat = self.calc_E()
        EoA = self.calc_trace_EoA(E)
        WoH = np.zeros((self.K, self.F, self.T), 'complex')
        upper = np.zeros((self.F, self.K, self.O), 'complex')
        lower = np.zeros((self.F, self.K, 1), 'complex')

        for k in range(self.K):
            W_k = self.W[:,k].reshape(-1,1)
            H_k = self.H[k,:].reshape(1,-1)
            WoH[k,:,:] = np.matmul(W_k, H_k)

        for f in range(self.F):
            WoH_f = WoH[:,f,:]
            EoA_f = EoA[f,:,:]
            upper[f,:,:] = np.matmul(WoH_f, EoA_f)

            lower[f,:] = np.matmul(WoH_f,xhat[f,:].reshape(-1,1))

        upper = np.sum(upper,axis=0)
        lower = np.sum(lower,axis=0)

        self.Q = np.multiply(self.Q, 1 + np.divide(upper, lower))


    def normaliseH(self):
        a_hat = np.sqrt(np.sum(np.power(self.H,2), axis=1)).reshape(self.K,1)
        self.H = self.H / a_hat
        self.W = self.W * a_hat.T

    def normaliseQ(self):
        b_hat = np.sqrt(np.sum(np.power(self.Q,2),axis=1)).reshape(self.K,1)
        self.Q = self.Q / b_hat
        self.W = self.W * b_hat.T

    def updateA(self):
        E, xhat = self.calc_E()
        QoW = np.zeros((self.O, self.F, self.K), 'complex')
        for o in range(self.O):
            for k in range(self.K):
                QoW[o,:,k] = np.multiply(self.Q[k,o], self.W[:,k])
        QoWH = np.matmul(QoW, self.H)
        left = np.zeros((self.O, self.F), 'complex')
        right = np.zeros((self.O, self.F, self.M, self.M), 'complex')
        for f in range(self.F):
            QoWH_f = QoWH[:,f,:]
            E_f = np.swapaxes(E[f,:,:,:],0,1)
            xhat_f = xhat[f,:].reshape(-1)
            left[:, f] = np.matmul(QoWH_f,xhat_f)
            right[:, f, :, :] = np.swapaxes(np.matmul(QoWH_f,E_f), 0, 1)
        A_hat = np.swapaxes(left.reshape(self.O, self.F, 1, 1) + right, 0, 1)

        for f in range(self.F):
            for o in range(self.O):
                LV, D, RV = linalg.svd(A_hat[f,o,:,:])
                D_hat = np.diag(D)
                D_hat[D_hat < 0] = 0
                A_hat_m = np.matmul(LV, np.matmul(D_hat, RV))

                self.A[f,o,:] = np.multiply(np.absolute(A_hat_m),
                                            np.exp(1j * np.angle(self.A[f,o,:,:])))
        return A_hat

    def cost_function(self, epoch):
        E, _ = self.calc_E()
        cost = np.linalg.norm(E)
        if self.DEBUG:
            print("Cost of epoch "+str(epoch)+" is: " + str(cost))
        return cost

    def run(self, epochs):
        if self.DEBUG:
            print("starting to run")

        self.cost_function(-1)
        for epoch in range(epochs):
            start1 = time.time()
            self.updateW()
            self.updateH()
            self.normaliseH()
            self.updateQ()
            self.normaliseQ()
            self.updateA()
            self.normaliseA()
            if self.DEBUG:
                print("epoch " + str(epoch) + " took " + str(time.time()-start1))
            self.cost_function(epoch)

    def reconstruct(self, x, sources=2):
        small_Q = self.Q[0,:].reshape(-1,1)
        b = np.zeros((sources,self.K))

        single_b = KMeans(n_clusters=sources, random_state=0).fit_predict(np.real(self.Q))
        for k in range(self.K):
            b[single_b[k], k] = 1

        BQ = np.matmul(b,self.Q)
        S = np.zeros((self.F, self.T, sources), dtype=self.A.dtype)
        S_full = np.zeros((self.F, self.T), dtype=self.A.dtype)
        for q in range(sources):
            for o in range(self.O):
                S[:, :, q] += np.matmul(np.multiply(BQ[q, o], self.W),self.H)
                S_full[:, :] += S[:, :, q]

        Y = np.zeros((self.F, self.T, sources), dtype=self.A.dtype)
        for q in range(sources):
            Y[:,:,q] = x * (S[:,:,q] / S_full)

        return Y


if __name__ == '__main__':
    SDR = 0
    SIR = 0
    SAR = 0
    files = 100    # files = ['/home/tinus/Workspace/corpus/TIMIT/TRAIN/DR1/FKFB0/SA1.WAV', '/home/tinus/Workspace/corpus/TIMIT/TRAIN/DR1/FDML0/SA1.WAV']
    isAC = False
    for index_files in range(files):

        # s1, s2 = load_files(corpus.experiment_files_voc())
        s1, s2 = load_files(corpus.experiment_files_MTS())
        length_speaker = len(s1)
        partition_speaker = 4
        room, locs = do_reverb(s1, s2)

        for i in range(0,length_speaker, int(length_speaker/partition_speaker)):
            half_s1 = s1[i:i+int(length_speaker/partition_speaker)]
            half_s2 = s2[i:i+int(length_speaker/partition_speaker)]
            half_m1 = room.mic_array.signals[0,i:i+int(length_speaker/partition_speaker)]
            half_m2 = room.mic_array.signals[1,i:i+int(length_speaker/partition_speaker)]

            Y1, Y2, X1, X2 = do_stft_2mic(half_m1, half_m2, half_s1, half_s2)

            signY1 = signum(Y1)
            signY2 = signum(Y2)
            magY1 = np.sqrt(np.abs(Y1))
            magY2 = np.sqrt(np.abs(Y2))
            X = np.asarray([signY1, signY2])
            M, F, T= X.shape
            X = X.reshape(F,T,M)
            bigX = np.zeros((F,T,M,M),'complex')
            # Xherm = np.conj(X.T)
            for f in range(F):
                for t in range(T):
                     bigX[f,t,:,:]= np.matmul(X[f,t,:], np.conj(X[f,t,:].T))

            ntf = SCM_CNTF(bigX, locs, K=3, max_az=5, max_th=5)
            ntf.run(4)

            Y = ntf.reconstruct(Y1)
            F,T,sources = Y.shape
            x_s = []
            for s in range(sources):
                new_x = np.real(istft(Y[:,:,s], win_length=256,hop_length=128))
                x_s.append(new_x)
                # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
            x_s = np.stack(x_s)
            x_stacked = np.vstack((half_s1[:x_s.shape[1]], half_s2[:x_s.shape[1]]))
            bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
            SDR += (bss[0][0] + bss[0][1])
            SIR += (bss[1][0] + bss[1][1])
            SAR += (bss[2][0] + bss[2][1])
    print(str(SAR/(partition_speaker*files)) + ","+ str(SDR/(partition_speaker*files)) + ","+str(SIR/(partition_speaker*files)) + ",")
