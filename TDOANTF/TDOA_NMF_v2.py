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

class SCM_NTF(object):
    def __init__(self, X, mic_locs, K=2,
                 fs=16000, win_size=1024, max_az=5, max_th=5, radius=1):
        self.X = X
        self.F, self.T, self.M, self.M = self.X.shape
        self.MM = self.M * self.M
        self.O = max_th * max_az
        self.X = X.reshape(self.F, self.T, self.MM)
        self.K = K
        self.radius = radius

        self.determine_A(max_az, max_th, fs, win_size, mic_locs)
        self.normalise_A()

        self.Q = np.random.random((self.K, self.O))
        self.W = np.random.random((self.F, self.K))
        self.H = np.random.random((self.K, self.T))


    def determine_A(self, max_azimuth, max_theta, fs, N, mic_locs):
        speed_of_sound = 344
        self.A = np.zeros((self.F, self.O, self.MM), dtype="complex")

        azimuth = 0
        theta = 0

        for o in range(self.O):
            if azimuth == max_azimuth:
                theta += 1
                azimuth = 0
            m = 0
            n = 0
            k_o = np.asarray([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(azimuth)), self.radius])
            for nm in range(self.MM):
                if m == self.M:
                    n += 1
                    m = 0
                tav_n = np.matmul(np.negative(k_o).T, mic_locs[n])/speed_of_sound
                tav_m = np.matmul(np.negative(k_o).T, mic_locs[m])/speed_of_sound

                tav_nm = tav_n - tav_m
                for i in range(F):
                    f_i = i * fs / N
                    self.A[i, o, nm] = np.exp(1j * 2 * np.pi * f_i * tav_nm)
                m += 1
            azimuth += 1

    def normalise_A(self):
        for f in range(self.F):
            for o in range(self.O):
                A_fo = self.A[f, o, :].reshape(self.M, self.M)
                self.A[f,o,:] = np.divide(A_fo, linalg.norm(A_fo)).reshape(self.MM)

    def calc_E(self):
        X_hat = np.zeros((self.F, self.T, self.MM), dtype="complex")
        x_hat = np.zeros((self.F, self.T), dtype="float")

        WH = np.matmul(self.W, self.H)

        for o in range(self.O):
            A_o = self.A[:,o,:].reshape(self.F, self.MM, 1)
            for k in range(self.K):
                x_hat_ko = self.Q[k,o] * WH
                x_hat += x_hat_ko

                X_hat += np.matmul(A_o, x_hat_ko.reshape(self.F, 1, self.T)).reshape(self.F, self.T, self.MM)

        E = self.X - X_hat

        return E, x_hat

    def cost_function(self, epoch):
        E, _ = self.calc_E()
        cost = np.linalg.norm(E)
        print("Cost of epoch "+str(epoch)+" is: " + str(cost))
        return cost


    def updateW(self):
        E, x_hat = self.calc_E()

        QH = np.multiply(np.sum(self.Q, axis=1), np.sum(self.H,axis=1))
        tmp_e = np.sum(E,axis=1).reshape(self.M,self.M, self.F)
        tmp_a = np.sum(self.A,axis=1).reshape(self.M,self.M, self.F)
        tr = np.abs(np.trace(np.multiply(tmp_e, tmp_a)))

        upper = np.matmul(QH.reshape(-1,1),
                          tr.reshape(1,-1)).T
        lower = np.matmul(QH.reshape(-1,1),
                          np.sum(x_hat,axis=1).reshape(1,-1)).T

        self.W = np.multiply(self.W, 1 + (np.divide(upper, lower)))

    def updateH(self):
        E, x_hat = self.calc_E()

        QW = (np.sum(self.Q, axis=1) * np.sum(self.W, axis=0)).reshape(1,self.K)
        tmp_e = np.sum(E, axis=0).reshape(self.T, self.MM)
        tmp_a = np.sum(np.sum(self.A, axis=1),
                       axis=0)

        tr = np.abs(np.trace(np.multiply(tmp_e,
                                         tmp_a).reshape(self.M,
                                                        self.M,
                                                        self.T)))
        upper = np.matmul(QW.reshape(-1, 1), tr.reshape(1, -1))
        lower = np.matmul(QW.reshape(-1, 1),
                          np.sum(x_hat, axis=0).reshape(1, -1))
        self.H = np.multiply(self.H, 1 + (np.divide(upper, lower)))

    def updateQ(self):
        E, x_hat = self.calc_E()

        WH = np.multiply(np.sum(self.W,axis=0),np.sum(self.H, axis=1))
        tmp_a = np.sum(self.A, axis=0).reshape(self.O, self.MM)
        tmp_e = np.sum(np.sum(E, axis=1),
                       axis=0)
        tr = np.abs(np.trace(np.multiply(tmp_e,
                                         tmp_a).reshape(self.M,
                                                        self.M,
                                                        self.O)))
        upper = np.matmul(WH.reshape(-1,1), tr.reshape(1,-1))
        lower = np.multiply(WH, np.sum(x_hat)).reshape(-1,1)
        self.Q = np.multiply(self.Q, 1+ np.divide(upper, lower))

    def updateA(self):
        E, x_hat = self.calc_E()
        WH = np.matmul(self.W, self.H)
        WHV = np.sum(np.multiply(WH, x_hat), axis=1).reshape(1,self.F)
        WHE = np.sum(np.multiply(WH.reshape(self.F,self.T,1),
                                 E),
                     axis=1).reshape(1,self.F,self.MM)
        Q_k = np.sum(self.Q,axis=0).reshape(self.O,1)
        left = np.matmul(Q_k, WHV).T
        right = np.einsum('kj,jnm->nkm',Q_k, WHE)
        A_hat = np.multiply(self.A, left.reshape(self.F, self.O, 1)+right)
        A_hat = A_hat.reshape(self.F, self.O, self.M, self.M)
        for f in range(self.F):
            for o in range(self.O):
                LV, D, RV = linalg.svd(A_hat[f,o,:,:])
                D_hat = np.diag(D)
                D_hat[D_hat < 0] = 0
                A_hat_m = np.matmul(LV, np.matmul(D_hat, RV))

                self.A[f,o,:] = np.multiply(np.absolute(A_hat_m.reshape(self.MM)),
                                            np.exp(1j * np.angle(self.A[f,o,:])))
        return A_hat

    def normaliseH(self):
        a_hat = np.sqrt(np.sum(np.power(self.H,2), axis=1)).reshape(self.K,1)
        self.H = self.H / a_hat
        self.W = self.W * a_hat.T

    def normaliseQ(self):
        b_hat = np.sqrt(np.sum(np.power(self.Q,2),axis=1)).reshape(self.K,1)
        self.Q = self.Q / b_hat
        self.W = self.W * b_hat.T

    def normaliseA(self):
        for f in range(self.F):
            for o in range(self.O):
                A_fo = self.A[f, o, :].reshape(self.M, self.M)
                self.A[f,o,:] = np.divide(A_fo, linalg.norm(A_fo)).reshape(self.MM)


    def run(self, epochs=10):
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

            self.cost_function(epoch)
            print("epoch " + str(epoch) + " took " + str(time.time()-start1))


if __name__ == '__main__':
    files = ['/home/teun/Workspace/Prive/TIMIT/TRAIN/DR1/FKFB0/SA1.WAV', '/home/teun/Workspace/Prive/TIMIT/TRAIN/DR1/FDML0/SA1.WAV']
    s1, s2 = load_files(files)
    room, locs = do_reverb(s1, s2)
    Y1, Y2, X1, X2 = do_stft(s1,s2,room)
    signY1 = signum(Y1)
    signY2 = signum(Y2)
    mag_sqrtY1 = np.sqrt(np.abs(Y1))
    mag_sqrtY2 = np.sqrt(np.abs(Y2))
    Y1_hat = np.multiply(mag_sqrtY1, signY1)
    Y2_hat = np.multiply(mag_sqrtY2, signY2)
    X = np.array([Y1_hat, Y2_hat]).T
    F,T,M = X.shape
    Xv1 = np.array([X[:,:,0] * X[:,:,0], X[:,:,0] * X[:,:,1], X[:,:,1] * X[:,:,0], X[:,:,1] * X[:,:,1]]).reshape(F,T,M,M)
    Xv2 = np.array([mag_sqrtY1, np.multiply(np.sqrt(np.abs(np.multiply(Y1,Y2))),(signum(np.multiply(Y1,Y2)))), np.multiply(np.sqrt(np.abs(np.multiply(Y2,Y1))),(signum(np.multiply(Y2,Y1)))), mag_sqrtY2]).reshape(F,T,M,M)
    print(Xv1.shape)
    print(Xv2.shape)
    print(Xv1[0,0])
    print(Xv2[0,0])

    ntf = SCM_NTF(Xv2, locs)
    ntf.run(10)
