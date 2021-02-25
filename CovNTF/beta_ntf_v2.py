import numpy as np
import scipy.linalg as linalg
from sklearn.cluster import KMeans

import time

# from utils import stft, istft, write_wav, read_wav
from utils_reverb import load_files, do_reverb, do_stft
import matplotlib.pyplot as plt

from librosa import istft

import mir_eval
import corpus

def signum(x):
    return x/np.abs(x)

class SpatCov_NTF(object):
    def __init__(self, X, mic_locs, K_partition=[3,3], J=2):

        self.X = X
        self.F, self.T, self.M, self.M = self.X.shape
        self.K = np.sum(K_partition)
        self.J = J
        self.MJ = self.M * self.J

        mix_psd = 0.5 * (np.mean(np.power(np.abs(self.X[:,:,0,0]),2) + np.power(np.abs(self.X[:,:,1,1]),2),axis=1))
        mix_psd = mix_psd.reshape((-1, 1))
        self.A = 0.5 * np.multiply(1.9 * np.abs(np.random.randn(self.M,self.MJ,self.F)) + 0.1 * np.ones((self.M,self.MJ,self.F)),signum(np.random.randn(self.M,self.MJ,self.F) + 1j *np.random.randn(self.M,self.MJ,self.F)))

        self.W = 0.5 * np.multiply(np.abs(np.random.randn(self.F,self.K)) + np.ones((self.F,self.K)), np.matmul(mix_psd, np.ones((1,self.K))))
        self.H = 0.5 * np.abs(np.random.randn(self.K,self.T)) + np.ones((self.K,self.T))
        self.Q = 0.5 * np.abs(np.random.randn(self.J,self.K)) + np.ones((self.J,self.K))
        self.sigma_b = np.matmul(mix_psd / 100, np.ones((1,self.T))).astype('complex')
        self.I = np.diag(np.ones((self.MJ)))

        self.O = np.ones((1,self.T))
        self.source_ind = []
        for j in range(self.J):
            self.source_ind.append(np.arange(0,int(self.K/self.J))+int(j*(self.K/self.J)))

        self.calculateV()

    def E_step(self):
        self.calculateV()
        sigma_s = self.calcSigmas()
        sigma_x = self.calcSigmaX(sigma_s)
        omega_s = self.calcOmegas(sigma_s, sigma_x)

        sigma_hat_xs = self.calcSigmahatxs(omega_s)
        sigma_hat_s = self.calcSigmahats(omega_s,sigma_s)
        self.calculateSigmab(sigma_x, sigma_hat_xs, sigma_hat_s)

        xi = self.calcxi(sigma_hat_s)

        return xi, sigma_hat_xs, sigma_hat_s

    def calculateSigmab(self, sigma_x, sigma_hat_xs, sigma_hat_s):
        for f in range(self.F):
            for t in range(self.T):
                Axs = np.matmul(self.A[:,:,f],
                                np.conj(np.transpose(sigma_hat_xs[:,:,f,t])))
                xsA = np.matmul(sigma_hat_xs[:,:,f,t],
                                np.conj(np.transpose(self.A[:,:,f])))
                AsA = np.matmul(np.matmul(self.A[:,:,f],
                                          sigma_hat_s[:,:,f,t]),
                                np.conj(np.transpose(self.A[:,:,f])))
                self.sigma_b[f,t] = 0.5 * np.trace(sigma_x[:,:,f,t] - \
                                    Axs - xsA + AsA)



    def calcxi(self, sigma_hat_s):
        xi = np.zeros((self.J, self.F, self.T), 'complex')

        for j in range(self.J):
            xi_mfn = np.zeros((self.M, self.F,self.T), 'complex')
            start_index = (j*self.I)
            end_index = (j+1) * self.I
            xi_i = 0
            for i in range(self.M):
                xi_mfn[xi_i, :, :] = sigma_hat_s[i,i,:,:]/self.M
                xi_i += 1
            xi[j,:,:] = np.sum(xi_mfn, axis=0)

        return xi


    def calcSigmahats(self, omega_s, sigma_s):
        sigma_hat_s = np.zeros((self.MJ, self.MJ, self.F, self.T),'complex')
        for f in range(self.F):
            for t in range(self.T):
                sigma_hat_s[:,:,f,t] = np.matmul(np.matmul(omega_s[:,:,f,t],
                                                           self.X[f,t,:,:]),
                                                 np.conj(omega_s[:,:,f,t].T)) + \
                                       np.matmul(self.I - np.matmul(omega_s[:,:,f,t],
                                                                    self.A[:,:,f]),
                                                 sigma_s[:,:,f,t])

        return sigma_hat_s

    def calcSigmahatxs(self, omega_s):
        sigma_hat_xs = np.zeros((self.M, self.MJ,self.F, self.T),'complex')
        for f in range(self.F):
            for t in range(self.T):
                sigma_hat_xs[:,:, f,t] = np.matmul(self.X[f,t,:,:],
                                                   np.conj(omega_s[:,:,f,t].T))

        return sigma_hat_xs

    def calcOmegas(self, sigma_s, sigma_x):
        Omega_s = np.zeros((self.MJ, self.M, self.F, self.T), 'complex')
        for f in range(self.F):
            for t in range(self.T):
                Omega_s[:,:,f,t] = np.matmul(np.matmul(sigma_s[:,:,f,t],
                                                       np.conj(self.A[:,:,f].T)),
                                             np.linalg.pinv(sigma_x[:,:,f,t]))
        return Omega_s

    def calcSigmaX(self, sigma_s):
        sigma_x = np.zeros((self.M, self.M, self.F, self.T), 'complex')

        for f in range(self.F):
            for t in range(self.T):
                sigma_x[:,:,f,t] = np.matmul(np.matmul(self.A[:,:,f],
                                                       sigma_s[:,:,f,t]),
                                             np.conj(self.A[:,:,f].T)) + \
                                   self.sigma_b[f,t]

        return sigma_x

    def calcSigmas(self):
        sigma_s = np.zeros((self.M,self.J,self.F,self.T), 'complex')
        for i in range(self.M):
            sigma_s[i,:,:,:] = self.V[:,:,:]
        temp_sigma_s = sigma_s.reshape((self.MJ, self.F, self.T))
        sigma_s = np.zeros((self.MJ,self.MJ,self.F,self.T), 'complex')
        for i in range(self.MJ):
            sigma_s[i,:,:,:] = temp_sigma_s[:,:,:]
        return sigma_s

    def calculateV(self):
        WoH = np.zeros((self.F, self.T, self.K), 'complex')
        for k in range(self.K):
            W_k = self.W[:,k].reshape(-1,1)
            H_k = self.H[k,:].reshape(1,-1)
            WoH[:,:,k] = np.matmul(W_k, H_k)

        self.V = np.zeros((self.J, self.F, self.T), 'complex')
        for j in range(self.J):
            self.V[j, :, :] = np.sum(np.multiply(WoH,self.Q[j,:].reshape(1,1,self.K)),axis=-1)

    def M_step(self, xi, sigma_hat_xs, sigma_hat_s):
        self.calculateA(sigma_hat_xs, sigma_hat_s)
        self.calculateQ(xi)
        self.calculateW(xi)
        self.calculateH(xi)

        self.normaliseA()
        self.normaliseQ()
        self.normaliseH()

    def calculateH(self, xi):
        self.calculateV()

        WoQ = np.zeros((self.J, self.K, self.F), 'complex')
        for j in range(self.J):
            for k in range(self.K):
                WoQ[j,k,:] = np.multiply(self.W[:,k], self.Q[j,k])

        upper = np.zeros((self.J, self.K, self.F, self.T), 'complex')
        lower = np.zeros((self.J, self.K, self.F, self.T), 'complex')

        for k in range(self.K):
            for j in range(self.J):
                upper[j,k,:,:] = np.multiply(np.multiply(WoQ[j,k,:].reshape(-1,1),
                                                         xi[j,:,:]),
                                             np.power(self.V[j,:,:], -2))
                lower[j,k,:,:] = np.multiply(WoQ[j,k,:].reshape(-1,1),
                                             np.power(self.V[j,:,:], -1))

        upper = np.sum(np.sum(upper, axis=-2), axis=0)
        lower = np.sum(np.sum(lower, axis=-2), axis=0)

        self.H = np.multiply(self.H, np.divide(upper, lower))

    def calculateW(self, xi):
        self.calculateV()
        HoQ = np.zeros((self.J,self.K,self.T),'complex')
        for j in range(self.J):
            for k in range(self.K):
                HoQ[j,k,:] = np.multiply(self.H[k,:],self.Q[j,k])

        upper = np.zeros((self.J, self.K, self.F, self.T), 'complex')
        lower = np.zeros((self.J, self.K, self.F, self.T), 'complex')

        for k in range(self.K):
            for j in range(self.J):
                upper[j,k,:,:] = np.multiply(np.multiply(HoQ[j,k,:],
                                                         xi[j,:,:]),
                                             np.power(self.V[j,:,:], -2))
                lower[j,k,:,:] = np.multiply(HoQ[j,k,:],
                                             np.power(self.V[j,:,:], -1))
        upper = np.sum(np.sum(np.swapaxes(upper,1,2), axis=-1), axis=0)
        lower = np.sum(np.sum(np.swapaxes(lower,1,2), axis=-1), axis=0)

        self.W = np.multiply(self.W, np.divide(upper, lower))

    def calculateQ(self, xi):
        self.calculateV()
        eta = 0.25
        WoH = np.zeros((self.F, self.T, self.K), 'complex')
        for k in range(self.K):
            W_k = self.W[:,k].reshape(-1,1)
            H_k = self.H[k,:].reshape(1,-1)
            WoH[:,:,k] = np.matmul(W_k, H_k)

        upper = np.zeros((self.J, self.K, self.F, self.T), 'complex')
        lower = np.zeros((self.J, self.K, self.F, self.T), 'complex')
        for k in range(self.K):
            for j in range(self.J):
                upper_first = np.multiply(WoH[:,:,k], xi[j,:,:])
                V_neg_sq = np.power(self.V[j,:,:]+eta, -2)
                upper[j,k,:,:] = np.multiply(upper_first, V_neg_sq)
                lower[j,k,:,:] = np.multiply(WoH[:,:,k],
                                             np.power(self.V[j,:,:], -1))

        upper = np.sum(np.sum(upper, axis=-1),axis=-1)
        lower = np.sum(np.sum(lower, axis=-1),axis=-1)

        self.Q = np.multiply(self.Q, np.divide(upper, lower))

    def calculateA(self, sigma_hat_xs, sigma_hat_s):
        self.A = np.zeros((self.M, self.MJ,self.F), 'complex')
        for f in range(self.F):
            for t in range(self.T):
                inv = np.linalg.pinv(sigma_hat_s[:,:,f,t])
                self.A[:,:,f] += np.matmul(sigma_hat_xs[:,:,f,t],
                                           inv)
    def normaliseA(self):
        for j in range(self.J):
            nonzero_f_ind = np.nonzero(self.A[:, j, :])
            self.A[:, j, nonzero_f_ind] = np.divide(self.A[:, j, nonzero_f_ind], signum(self.A[:,j,nonzero_f_ind]))
            # self._A[0, j, nonzero_f_ind] = np.divide(self.A[0, j, nonzero_f_ind], signum(self.A[0,j,nonzero_f_ind]))

            A_scale = np.sum(np.power(np.abs(self.A[:,j,:]),2), axis=0)
            self.A[:, j,:] = np.divide(self.A[:, j,:], np.tile(np.sqrt(A_scale).reshape(1,-1),(self.M,1)))
            self.W[:,self.source_ind[j]] = np.multiply(self.W[:,self.source_ind[j]], np.matmul(A_scale.reshape(-1,1),np.ones((1,len(self.source_ind[j])))))

    def normaliseQ(self):
        scale = np.sum(self.Q, axis=0)
        self.Q = np.multiply(self.Q, np.tile(np.power(scale,-1),(self.J,1)))
        self.W = np.multiply(self.W, np.tile(scale,(self.F,1)))

    def normaliseH(self):
        scale = np.sum(self.W, axis=0).reshape(1,-1)

        self.W = np.multiply(self.W, np.tile(np.power(scale,-1),(self.F,1)))
        self.H = np.multiply(self.H, np.tile(scale.transpose(),(1,self.T)))

    def run(self, epochs=100):
        print("Running")
        for epoch in range(epochs):
            # print(epoch)
            xi, sigma_hat_xs, sigma_hat_s = self.E_step()
            self.M_step(xi, sigma_hat_xs, sigma_hat_s)
            # print(epoch)

    def reconstruct(self, Xb):
        self.calculateV()
        Y = np.zeros((self.M, self.J, self.F, self.T), 'complex')
        for f in range(self.F):
            for t in range(self.T):
                RV = np.zeros((self.M, self.M), 'complex')
                for j in range(self.J):
                    start_index = (j*self.M)
                    end_index = (j+1) * self.M
                    R_i = np.matmul(self.A[:,start_index:end_index,f],
                                    np.conj(self.A[:,start_index:end_index,f].T))
                    RV += np.multiply(R_i, self.V[j,f,t])
                RV = np.linalg.pinv(RV)
                for j in range(self.J):
                    start_index = (j*self.M)
                    end_index = (j+1) * self.M
                    R_i = np.matmul(self.A[:,start_index:end_index,f],
                                    np.conj(self.A[:,start_index:end_index,f].T))
                    Y[:,j,f,t] = np.matmul(np.matmul(np.multiply(R_i,
                                                                 self.V[j,f,t]),
                                                     RV),
                                           Xb[:,f,t])


        return Y


if __name__ == '__main__':
    # SDR = [-3946.01136493, -3816.54215396]
    # SIR = [407.75139811, 597.85980611]
    # SAR = [-3469.03607208, -3414.63520819]
    SDR = 0
    SIR = 0
    SAR = 0
    files = 100    # files = ['/home/tinus/Workspace/corpus/TIMIT/TRAIN/DR1/FKFB0/SA1.WAV', '/home/tinus/Workspace/corpus/TIMIT/TRAIN/DR1/FDML0/SA1.WAV']
    for index_files in range(files):

        s1, s2 = load_files(corpus.experiment_files_MTS())
        # files = ['/home/teun/Workspace/Prive/TIMIT/TRAIN/DR1/FKFB0/SA1.WAV', '/home/teun/Workspace/Prive/TIMIT/TRAIN/DR1/FDML0/SA1.WAV']
        # s1, s2 = load_files(files)
        room, locs = do_reverb(s1, s2)
        Y1, Y2, X1, X2 = do_stft(s1,s2,room)
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

        ntf = SpatCov_NTF(bigX, locs, K_partition=[5,5], J=2)
        ntf.run(4)
        X = np.asarray([Y1, Y2])
        Y = ntf.reconstruct(X)
        m,sources,F,T = Y.shape
        x_s = []
        for s in range(sources):
            new_x = np.real(istft(Y[0,s,:,:], win_length=256,hop_length=128))
            x_s.append(new_x)
            # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
        x_s = np.stack(x_s)
        x_stacked = np.vstack((s1[:x_s.shape[1]], s2[:x_s.shape[1]]))
        bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
        SDR += bss[0]
        SIR += bss[1]
        SAR += bss[2]
        x_s = []
        for s in range(sources):
            new_x = np.real(istft(Y[1,s,:,:], win_length=256,hop_length=128))
            x_s.append(new_x)
            # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
        x_s = np.stack(x_s)
        x_stacked = np.vstack((s1[:x_s.shape[1]], s2[:x_s.shape[1]]))
        bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
        SDR += bss[0]
        SIR += bss[1]
        SAR += bss[2]
        print("DEBUG:--"+str(index_files))
        print("DEBUG:--"+str(SAR) + ","+ str(SDR) + ","+str(SIR))
        print("DEBUG:--"+str(SAR/(2*index_files)) + ","+ str(SDR/(2*index_files)) + ","+str(SIR/(2*index_files)))
    print(str(SAR/(2*files)) + ","+ str(SDR/(2*files)) + ","+str(SIR/(2*files)) + ",")
