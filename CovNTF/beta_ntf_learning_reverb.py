import numpy as np
import scipy.linalg as linalg
from sklearn.cluster import KMeans

import time

# from utils import stft, istft, write_wav, read_wav
from utils_reverb import load_files, do_reverb, do_stft
import matplotlib.pyplot as plt

import mir_eval

def signum(x):
    return x/np.abs(x)

class SpatCov_NTF_H1(object):
    def __init__(self, X, mic_locs, V, K_partition=[3,3], J=2):

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

        self.V = V

    def E_step(self):
        sigma_s = self.calcSigmas()
        sigma_x = self.calcSigmaX(sigma_s)
        omega_s = self.calcOmegas(sigma_s, sigma_x)

        sigma_hat_xs = self.calcSigmahatxs(omega_s)
        sigma_hat_s = self.calcSigmahats(omega_s,sigma_s)
        self.calculateSigmab(sigma_x, sigma_hat_xs, sigma_hat_s)

        return sigma_hat_xs, sigma_hat_s

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
                                             np.linalg.inv(sigma_x[:,:,f,t]))
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


    def M_step(self, sigma_hat_xs, sigma_hat_s):
        self.calculateA(sigma_hat_xs, sigma_hat_s)

        self.normaliseA()

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

    def run(self, epochs=100):
        print("Running")

        self.E_step()


    def reconstruct(self, Xb):
        self.calculateV()
        Y = np.zeros((self.M, self.J, self.F, self.T), 'complex')
        for f in range(self.F):
            for t in range(self.T):
                RV = np.zeros((self.M, self.M))
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

class SpatCov_NTF_Hs(SpatCov_NTF_H1):
    def __init__(self, X, mic_locs, V, K_partition=[3,3], J=2):
        super(SpatCov_NTF_Hs, self).__init__(X,mic_locs,V, K_partition, J)

    def calculateA(self, bar_Rxx, bar_Rxs, bar_Rsx, bar_Rss, s_value=0.9999):
        G_xyxy = np.array([[bar_Rxx, bar_Rxs],[bar_Rsx, bar_Rss]]).reshape(2*self.F, 2*self.T)

        U, s, V = linalg.svd(G_xyxy)

        total_s = np.sum(s)
        summed_s = []
        inter_total_s = 0
        for val_s in s:
            inter_total_s += val_s
            summed_s.append(inter_total_s/total_s)
        # print(summed_s[0])
        indices = np.array(np.where(np.array(summed_s) < s_value))
        p_value = indices[0,-1]
        U_p = U[:self.F,:p_value]
        V_p = V[:p_value,self.T:]

        return np.matmul(U_p, linalg.pinv(V_p).T)

    def E_step(self):
        print("correct E")
        sigma_s = self.calcSigmas()
        sigma_x = self.calcSigmaX(sigma_s)
        omega_s = self.calcOmegas(sigma_s, sigma_x)

        sigma_hat_xs = self.calcSigmahatxs(omega_s)
        sigma_hat_sx = self.calcSigmahatsx(omega_s)
        sigma_hat_s = self.calcSigmahats(omega_s,sigma_s)
        self.calculateSigmab(sigma_x, sigma_hat_xs, sigma_hat_s)
        self.A = np.zeros((self.M, self.MJ, self.F), 'complex')
        for m1 in range(self.M):
            for m2 in range(self.M):
                sigma_x_mm = sigma_x[m1, m2,:,:]
                for mj1 in range(self.MJ):
                    sigma_hat_xs_mmj = sigma_hat_xs[m1, mj1,:,:]
                    sigma_hat_sx_mmj = sigma_hat_sx[m1, mj1,:,:]
                    for mj2 in range(self.MJ):
                        sigma_hat_s_mjmj = sigma_hat_s[mj1, mj2, :, :]
                        temp_A_mmj =  self.calculateA(sigma_x_mm,
                                                   sigma_hat_xs_mmj,
                                                   sigma_hat_sx_mmj,
                                                   sigma_hat_s_mjmj)
                        self.A[m1, mj1, :] += np.sum(temp_A_mmj, axis=-1)

        self.normaliseA()



    def calcSigmahatxs(self, omega_s):
        sigma_hat_xs = np.zeros((self.M, self.MJ,self.F, self.T),'complex')
        for f in range(self.F):
            for t in range(self.T):
                sigma_hat_xs[:,:, f,t] = np.matmul(self.X[f,t,:,:],
                                                   np.conj(omega_s[:,:,f,t].T))

        return sigma_hat_xs

    def calcSigmahatsx(self, omega_s):
        sigma_hat_sx = np.zeros((self.M, self.MJ,self.F, self.T),'complex')

        for f in range(self.F):
            for t in range(self.T):
                temp_sx = np.matmul(omega_s[:,:,f,t],
                                    np.conj(self.X[f,t,:,:].T))
                sigma_hat_sx[:,:, f,t] = np.swapaxes(temp_sx, 0, 1)

        return sigma_hat_sx

if __name__ == '__main__':
    files = ['/home/teun/Workspace/Prive/TIMIT/TRAIN/DR1/FKFB0/SA1.WAV', '/home/teun/Workspace/Prive/TIMIT/TRAIN/DR1/FDML0/SA1.WAV']
    s1, s2 = load_files(files)
    room, locs = do_reverb(s1, s2)
    Y1, Y2, X1, X2 = do_stft(s1,s2,room)

    X = np.asarray([Y1, Y2])
    M, F, T= X.shape
    X = X.reshape(F,T,M)
    bigX = np.zeros((F,T,M,M),'complex')
    # Xherm = np.conj(X.T)
    for f in range(F):
        for t in range(T):
             bigX[f,t,:,:]= np.matmul(X[f,t,:], np.conj(X[f,t,:].T))

    X = np.asarray([X1, X2])
    ntf = SpatCov_NTF_H1(bigX, locs, X, K_partition=[3,3], J=2)
    ntf.run(5)

    ntf = SpatCov_NTF_Hs(bigX, locs, X, K_partition=[3,3], J=2)
    ntf.run(5)
