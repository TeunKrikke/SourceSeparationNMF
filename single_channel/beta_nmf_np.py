"""
 Contains an error which makes the soundlevels go down.
"""

import time
import numpy as np

from librosa import load, stft, istft, resample
from librosa.output import write_wav
from sklearn.cluster import MiniBatchKMeans, FeatureAgglomeration
from sklearn import datasets

# import matplotlib.pyplot as plt

import mir_eval

import corpus

class beta_NMF(object):
    """docstring for beta_NMF"""
    def __init__(self, frequencies, time_steps, sources, X):
        super(beta_NMF, self).__init__()
        self._frequencies = frequencies
        self._time_steps = time_steps
        self._sources = sources
        self._epochs = 10
        self._V = X
        self._W = np.asarray(np.random.rand(self._frequencies, self._sources)+np.ones((self._frequencies, self._sources)))
        self._H = np.asarray(np.random.rand(self._sources, self._time_steps)+np.ones((self._sources, self._time_steps)))
        index = 0

    def train(self):

        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W,self._H)
            self._H = np.multiply(self._H, np.dot(np.transpose(self._W), np.multiply(self._V, np.power(V_hat,-2))) / np.dot(np.transpose(self._W), np.power(V_hat, -1)))

            V_hat = np.dot(self._W,self._H)
            self._W = np.multiply(self._W, np.dot(np.multiply(self._V, np.power(V_hat,-2)), np.transpose(self._H)) / np.dot(np.power(V_hat,-1), np.transpose(self._H)))

            V_hat = np.dot(self._W,self._H)

            cost = np.sum(self._V/V_hat - np.log(self._V/V_hat)) - 1

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._frequencies,1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._time_steps,1)))
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H

    def reconstruct(self,k):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(self._W,self._H)
        W = self._W[:,k].reshape((-1,1))
        H = self._H[k,:].reshape((1,-1))
        return np.multiply((np.dot(W,H)/V_hat), self._V)

    def reconstruct_with_Z(self,k,Z):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(self._W,self._H)
        W = self._W
        H = np.multiply(self._H,(Z == k).astype(int).reshape(-1,1))

        # print(H.shape)
        return np.multiply((np.dot(W,H)/V_hat), self._V)

    def reconstruct_with_Z_as_H(self,K,Z):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(self._W,self._H)
        W = K
        H = Z

        # print(H.shape)
        return np.multiply((np.dot(W,H)/V_hat), self._V)

    def reconstruct_with_Z_t(self,k,Z):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(self._W,self._H)
        W = self._W

        H = np.multiply(self._H,(Z == k).astype(int))
        # print(H.shape)
        return np.multiply((np.dot(W,H)/V_hat), self._V)

def main():
    files = 10
    for sources in range(50, 1000, 50):
        SAR1 = 0
        SDR1 = 0
        SIR1 = 0
        SAR2 = 0
        SDR2 = 0
        SIR2 = 0
        SAR3 = 0
        SDR3 = 0
        SIR3 = 0
        for index_files in range(files):
            f1, f2 = corpus.experiment_files_voc()
            x1, sr = load(f1)
            x2, sr = load(f2)
            nsamp = max(len(x1), len(x2))+1 # determine which file is longest and save that lenght
            # make both files even length by zero padding
            x1 = np.pad(x1, (0,nsamp-len(x1)), mode='constant', constant_values=0)
            x2 = np.pad(x2, (0,nsamp-len(x2)), mode='constant', constant_values=0)
            # x = (x1/2) + (x2/2)
            x = x1 + x2
            # x = resample(x, 192000, 16000)
            X = stft(x, win_length=256,hop_length=128, n_fft=1024)
            X = np.abs(X)**2
            F,T = X.shape
            write_wav('np_beta_separated_a.wav', x, sr)
            frequencies, time_steps = X.shape
            # sources = 300
            nmf = beta_NMF(frequencies, time_steps, sources, X)
            W, H = nmf.train()
            clusters = 2
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=1,
                              n_init=10, max_no_improvement=50, verbose=0)

            Z = mbk.fit_predict(H)

            x_s = []
            for s in range(clusters):
                new_x = np.real(istft(nmf.reconstruct_with_Z(s,Z), win_length=256,hop_length=128))
                x_s.append(new_x)
                # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
            x_s = np.stack(x_s)
            x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
            bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
            SDR1 += bss[0]
            SIR1 += bss[1]
            SAR1 += bss[2]

            agglo = FeatureAgglomeration(n_clusters=2)
            Z = agglo.fit_transform(H.T).T
            agglo = FeatureAgglomeration(n_clusters=2)
            S = agglo.fit_transform(W)
            x_s = []
            for s in range(clusters):
                new_x = np.real(istft(nmf.reconstruct_with_Z_as_H(S[:,s].reshape((-1,1)),Z[s,:].reshape(1,-1)), win_length=256,hop_length=128))
                x_s.append(new_x)
                # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
            x_s = np.stack(x_s)
            x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
            bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
            SDR3 += bss[0]
            SIR3 += bss[1]
            SAR3 += bss[2]

            # Z = np.zeros(H.shape)
            # for t in range(H.shape[1]):
            #     Z[:,t] = mbk.fit_predict(H[:,t].reshape(1,-1))
            #
            # x_s = []
            # for s in range(clusters):
            #     new_x = np.real(istft(nmf.reconstruct_with_Z_t(s,Z), win_length=256,hop_length=128))
            #     x_s.append(new_x)
            #     # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
            # x_s = np.stack(x_s)
            # x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
            # bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
            # SDR2 += bss[0]
            # SIR2 += bss[1]
            # SAR2 += bss[2]
        print("beta_NMF {} basis".format(sources))
        print("total cluster")
        print(np.sum(SAR1) / (2*files))
        print(np.sum(SDR1) / (2*files))
        print(np.sum(SIR1) / (2*files))
        # print("cluster per timestep")
        # print(np.sum(SAR2) / (2*files))
        # print(np.sum(SDR2) / (2*files))
        # print(np.sum(SIR2) / (2*files))
        print("cluster using agglo")
        print(np.sum(SAR3) / (2*files))
        print(np.sum(SDR3) / (2*files))
        print(np.sum(SIR3) / (2*files))


if __name__ == '__main__':
    # digits = datasets.load_digits()
    # images = digits.images
    # print(images.shape)
    # X = np.reshape(images, (len(images), -1))
    # print(X.shape)
    # agglo = FeatureAgglomeration(n_clusters=32)
    # agglo.fit(X)
    #
    #
    #
    # X_reduced = agglo.transform(X)
    # print(X_reduced.shape)
    main()
    # x, sr = load('/home/tinus/Workspace/tensorflow_ws/Source_separation/NMF/H_T_200Mic1NNE.wav')
    # x, sr = load('/home/tinus/Workspace/tensorflow_ws/Source_separation/NMF/H_T_200Mic1NE.wav')
    # x, sr = load('/home/tinus/Workspace/tensorflow_ws/Source_separation/NMF/H_T_200Mic1NN.wav')
    # x, sr = load('/home/tinus/Workspace/tensorflow_ws/Source_separation/NMF/H_T_200Mic1O.WAV')

    # x1, sr = load('/home/tinus/Workspace/corpus/data/S0001.wav')
    # x2, sr = load('/home/tinus/Workspace/corpus/data/S0006.wav')
    # nsamp = max(len(x1), len(x2))+1 # determine which file is longest and save that lenght
    # # make both files even length by zero padding
    # x1 = np.pad(x1, (0,nsamp-len(x1)), mode='constant', constant_values=0)
    # x2 = np.pad(x2, (0,nsamp-len(x2)), mode='constant', constant_values=0)
    # # x = (x1/2) + (x2/2)
    # x = x1 + x2
    # # x = resample(x, 192000, 16000)
    # X = stft(x, win_length=256,hop_length=128, n_fft=1024)
    # X = np.abs(X)**2
    # F,T = X.shape
    # write_wav('np_beta_separated_a.wav', x, sr)
    # frequencies, time_steps = X.shape
    # sources = 300
    # nmf = beta_NMF(frequencies, time_steps, sources, X)
    # W, H = nmf.train()
    # clusters = 2
    # mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=1,
    #                   n_init=10, max_no_improvement=50, verbose=0)
    #
    # Z = mbk.fit_predict(H)
    #
    # x_s = []
    # for s in range(clusters):
    #     new_x = np.real(istft(nmf.reconstruct_with_Z(s,Z), win_length=256,hop_length=128))
    #     x_s.append(new_x)
    #     # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
    # x_s = np.stack(x_s)
    # x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
    # bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
    # SDR = bss[0]
    # SIR = bss[1]
    # SAR = bss[2]
    # print("beta_NMF {} basis".format(sources))
    # print("total cluster")
    # print(np.sum(SAR) / (2))
    # print(np.sum(SDR) / (2))
    # print(np.sum(SIR) / (2))
    #
    # Z = np.zeros(H.shape)
    # for t in range(H.shape[1]):
    #     Z[:,t] = mbk.fit_predict(H[:,t].reshape(-1,1))
    #
    # x_s = []
    # for s in range(clusters):
    #     new_x = np.real(istft(nmf.reconstruct_with_Z_t(s,Z), win_length=256,hop_length=128))
    #     x_s.append(new_x)
    #     # write_wav('tests/voc_2basis_{}.wav'.format(s), new_x, sr)
    # x_s = np.stack(x_s)
    # x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
    # bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
    # SDR = bss[0]
    # SIR = bss[1]
    # SAR = bss[2]
    # print("cluster per timestep")
    # print(np.sum(SAR) / (2))
    # print(np.sum(SDR) / (2))
    # print(np.sum(SIR) / (2))

    # colors = ['r', 'g', 'b', 'y', 'c']
    # mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    # # for k, col in zip(range(clusters), colors):
    # #     cluster_center = mbk_means_cluster_centers[k]
    # #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    # #             markeredgecolor='k', markersize=6)
    #
    #
    # for k in range(clusters):
    #     cluster_center = mbk_means_cluster_centers[k]
    #     plt.plot(cluster_center[0], cluster_center[1], 'o',
    #             markeredgecolor='k', markersize=6)
    #
    # plt.show()


    # for s in range(sources):
    #     new_x = np.real(istft(nmf.reconstruct(s), win_length=256,hop_length=128))
    #     write_wav('sources/np_beta_separated_{}.wav'.format(s), new_x, sr)
