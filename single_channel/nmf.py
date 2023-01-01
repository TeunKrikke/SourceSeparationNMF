'''
Theano implementation for beta-divergence nonnegative matrix factorisation. This uses multiple cost and associated update functions (among others Itakura-Saito and Cauchy).
'''

from utils import create_mixture, do_STFT_on_data
import corpus

import time
import numpy as np
import theano as th
from theano import tensor as T

from librosa import load, stft, istft
from librosa.output import write_wav

from update_rules import update_h_beta, update_w_beta, update_w_cauchy, update_h_cauchy
from cost import cost_is, cost_cau, cost_euc, cost_kl
from utils import convolution, shift
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt

import mir_eval

th.config.optimizer = 'None'
th.config.exception_verbosity = 'high'


class NMF(object):
    """docstring for NMF"""
    def __init__(self, frequencies, time_steps, sources, X, conv=False, beta=2):
        """
        NMF constructor

        The W matrix shows the frequencies per source
        The H matrix shows the activation per time step per source. Ie how does the time_step belongs to source X and source Y

        Keyword arguments:
        frequencies -- the number of frequencies we want to approximate
        time_steps -- length of the file we want to approximate
        sources -- the number of sources we want to recognise
        X -- the magnitude of the original signal
        convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
        """
        super(NMF, self).__init__()
        self._DEBUG = True
        self._frequencies = frequencies
        self._time_steps = time_steps
        self._sources = sources
        self._epochs = 900 # the number of epochs to run for
        self._V = X
        self._T = 0
        self._beta = beta

        if conv and not beta == 2: # in the case of convolution the W matrix has an extra dimension
            self._T = 5
            self._W = th.shared(value=np.asarray(np.random.rand(self._frequencies, self._sources, self._T)+np.ones((self._frequencies, self._sources, self._T)), dtype=th.config.floatX), name="W", borrow=True)
        else:# otherwise the matrix is just 2D frequencies x sources
            self._W = th.shared(value=np.asarray(np.random.rand(self._frequencies, self._sources)+np.ones((self._frequencies, self._sources)), dtype=th.config.floatX), name="W", borrow=True)

        self._H = th.shared(value=np.asarray(np.random.rand(self._sources, self._time_steps)+np.ones((self._sources, self._time_steps)), dtype=th.config.floatX), name="H", borrow=True)
        index = T.lscalar()
        X = T.fmatrix()
        self.reconstruct_func = th.function(inputs=[index, X],
                                   outputs=self.reconstruct(self._W, self._H, X, index, conv),
                                   name="reconstruct",
                                   allow_input_downcast=True)



    def train(self, cost, update_w, update_h, convolution=False, norm_W=0, norm_H=0, beta=0):
        """
        Train the NMF

        Keyword arguments:
        cost -- the cost function
        update_w -- the rule for updating the W matrix
        update_h -- the update rule for the H marix
        convolution -- if we want to do convolution we need to add an extra dimension to the W matrix. (default False)
        norm_W -- normalise the W matrix with L1 norm (1) or L2 norm (2) (default 0)
        norm_H -- normalise the H matrix with L1 norm (1) or L2 norm (2) (default 0)
        beta -- the beta parameter that determines which update rule to use Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
        """

        # the H train function
        self.train_h = th.function(inputs=[],
                                       outputs=[],
                                       updates={self._H: update_h(self._V, self._W, self._H, beta, convolution, norm_H, self._T, self._time_steps)},
                                       name="train_H",
                                       allow_input_downcast=True)
        # the W train function
        self.train_w = th.function(inputs=[],
                                       outputs=[],
                                       updates={self._W: update_w(self._V, self._W, self._H, beta, convolution, norm_W, self._T, self._frequencies)},
                                       name="train_W",
                                       allow_input_downcast=True)
        # the cost function
        self.cost_func = th.function(inputs=[],
                                   outputs=cost(self._V, self._W, self._H, self._frequencies, self._time_steps, convolution),
                                   name="cost",
                                   allow_input_downcast=True)

        for epoch in range(self._epochs):
            tick = time.time()

            self.train_h()

            self.train_w()

            # scale both matrices
            scale = T.sum(self._W, axis=0)
            self._W = self._W * T.tile(T.pow(scale,-1),(self._frequencies,1))
            self._H = self._H * T.transpose(T.tile(T.pow(scale,-1),(self._time_steps,1)))
            if self._DEBUG:
                print ('NMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, self.cost_func()))


        # return train_func
    def get_W_H(self):
        return self._W.eval(), self._H.eval(), self._V
    def reconstruct(self, W, H, X, k, conv=False):
        """
        Reconstruct a source by applying a Wiener filter

        Keyword arguments
        W -- the frequency matrix F x K
        H -- the activation matrix per timestep K x N
        X -- the original input matrix which is NOT the magnitude F x N
        k -- the source index we want to reconstruct
        """
        if conv and not self._beta == 2:
            V_hat = convolution(W, H) # reconstruct the approximation of V
            W = W[:,k, :].reshape((-1,1, self._T)) # get a single column from the W matrix
            H = H[k,:].reshape((1,-1)) # get a single row from the H matrix
            V_k = T.zeros(V_hat.shape)
            for t in range(self._T):
                V_k = V_k + T.dot(W[:,:,t].reshape((-1,1)), shift(H,t))

            return T.mul((V_k/V_hat),X) # apply the Wiener filter to X
        else:
            V_hat = th.dot(W, H) # reconstruct the approximation of V
            W = W[:,k].reshape((-1,1)) # get a single column from the W matrix
            H = H[k,:].reshape((1,-1)) # get a single row from the H matrix
            return T.mul((T.dot(W,H)/V_hat),X) # apply the Wiener filter to X



# def kmeans(X, cluster_num, numepochs, learningrate=0.01, batchsize=100, verbose=True):
# 	'''
# 		klp_kmeans based NUMPY, better for small scale problems
# 		inherited from http://www.iro.umontreal.ca/~memisevr/code.html
#       Error in casting to float from complex in line 192
# 	'''
#
# 	rng = np.random
# 	W =rng.randn(cluster_num, X.shape[1])
# 	X2 = (X**2).sum(1)[:, None]
# 	for epoch in range(numepochs):
# 	    for i in range(0, X.shape[0], batchsize):
# 	        D = -2*np.dot(W, X[i:i+batchsize,:].T) + (W**2).sum(1)[:, None] + X2[i:i+batchsize].T
# 	        S = (D==D.min(0)[None,:]).astype("float").T
# 	        W += learningrate * (np.dot(S.T, X[i:i+batchsize,:]) - S.sum(0)[:, None] * W)
# 	    if verbose:
# 	        print "epoch", epoch, "of", numepochs, " cost: ", D.min(0).sum()
# 	return W

def reconstruct_with_Z(k,Z, W, H, V):
    # V_hat = th.dot(W, H)
    # W = W[:,k].reshape((-1,1))
    # H = H[k,:].reshape((1,-1))
    # return T.mul((T.dot(W,H)/V_hat),V)
    V_hat = np.dot(W,H)

    H_k = np.multiply(H,(Z == k).astype(int).reshape(-1,1))
    return np.multiply((np.dot(W,H_k)/V_hat), V)

# if __name__ == '__main__':
#     # x, sr = load('Hannah_teun_317Mic18.WAV')
#     # x, sr = load('H_T_200Mic1O.WAV')
#     x1, sr = load('/home/tinus/Workspace/corpus/data/S0001.wav')
#     x2, sr = load('/home/tinus/Workspace/corpus/data/S0006.wav')
#     # x = (x1/2) + (x2/2)
#     x = (x1) + (x2)
#     X = stft(x, win_length=256,hop_length=128, n_fft=1024)
#
#     V = np.abs(X)**2
#
#     frequencies, time_steps = X.shape
#     sources = 50
#     nmf = NMF(frequencies, time_steps, sources, V)
#     train = nmf.train(cost_is, update_w_beta, update_h_beta)
#     # train = nmf.train(cost_cau, update_w_cauchy, update_h_cauchy)
#     W, H, V = nmf.get_W_H()
#
#     clusters = 2
#     mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=sources,
#                       n_init=10, max_no_improvement=50, verbose=0)
#
#     Z = mbk.fit_predict(H)
#
#     # colors = ['r', 'g', 'b', 'y', 'c']
#     # mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
#     # # for k, col in zip(range(clusters), colors):
#     # #     cluster_center = mbk_means_cluster_centers[k]
#     # #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#     # #             markeredgecolor='k', markersize=6)
#     #
#     # for k in range(clusters):
#     #     cluster_center = mbk_means_cluster_centers[k]
#     #     plt.plot(cluster_center[0], cluster_center[1], 'o',
#     #             markeredgecolor='k', markersize=6)
#     #
#     # plt.show()
#     x_s = []
#     for s in range(clusters):
#         new_x = np.real(istft(reconstruct_with_Z(s, Z, W, H, V), win_length=256,hop_length=128))
#         x_s.append(new_x)
#         write_wav('tests/normal_cau_separated_{}.wav'.format(s), new_x, sr)
#     x_s = np.stack(x_s)
#     x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
#     bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
#     print(bss)
#
#
#     # for s in range(sources):
#     #     new_x = np.real(istft(nmf.reconstruct_func(s, X), win_length=256,hop_length=128))
#     #     write_wav('normal_separated_{}.wav'.format(s), new_x, sr)
if __name__ == '__main__':
    SDR = 0  # signal to distortion ratio over all files summed
    SAR = 0  # signal to artifact ratio over all files summed
    SIR = 0  # signal to inference ratio over all files summed

    for i in range(100):
        corpus_train = corpus.experiment_files_voc
        mix, x1, x2 = create_mixture(corpus_train)
        X, _, _ = do_STFT_on_data(mix, x1, x2)
        X = X.T

        V = np.abs(X)**2
        frequencies, time_steps = X.shape
        sources = 50
        nmf = NMF(frequencies, time_steps, sources, V)
        train = nmf.train(cost_is, update_w_beta, update_h_beta)

        W, H, V = nmf.get_W_H()

        clusters = 2
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=sources,
                          n_init=10, max_no_improvement=50, verbose=0)

        Z = mbk.fit_predict(H)
        x_s = []
        for s in range(clusters):
            new_x = np.real(istft(reconstruct_with_Z(s, Z, W, H, V), win_length=256,hop_length=128))
            x_s.append(new_x)

        x_s = np.stack(x_s)
        x_stacked = np.vstack((x1[:x_s.shape[1]], x2[:x_s.shape[1]]))
        bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
        SDR += bss[0]
        SIR += bss[1]
        SAR += bss[2]

    print(np.sum(SAR) / 200)
    print(np.sum(SDR) / 200)
    print(np.sum(SIR) / 200)
