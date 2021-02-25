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
from update_and_cost_func import update_H, update_W
from update_and_cost_func import update_H_CA, update_W_CA
from update_and_cost_func import cost as cost_fn

class beta_NMF(object):
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=0):
        super(beta_NMF, self).__init__()
        self._epochs = epochs
        self._debug = debug
        self._V = X
        self._W = W
        self._H = H
        index = 0
        self._beta = beta

    def train(self):
        pass

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


class Unsupervised_NMF(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, frequencies, time_steps, sources, X, epochs=1000,
                 debug=False, beta=0):
        self._frequencies = frequencies
        self._time_steps = time_steps
        W = np.asarray(np.random.rand(frequencies, sources) + np.ones((frequencies, sources)))
        H = np.asarray(np.random.rand(sources, time_steps) + np.ones((sources, time_steps)))
        super(Unsupervised_NMF, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):

        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W, self._H)
            self._H = update_H(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            self._W = update_W(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            cost = cost_fn(self._V, V_hat, self._beta)

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._frequencies,1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._time_steps,1)))
            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H

class Unsupervised_CA_NMF(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, frequencies, time_steps, sources, X, epochs=1000,
                 debug=False, beta=99):
        self._frequencies = frequencies
        self._time_steps = time_steps
        W = np.asarray(np.random.rand(frequencies, sources) + np.ones((frequencies, sources)))
        H = np.asarray(np.random.rand(sources, time_steps) + np.ones((sources, time_steps)))
        super(Unsupervised_CA_NMF, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):

        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W, self._H)
            self._H = update_H_CA(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            self._W = update_W_CA(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            cost = cost_fn(self._V, V_hat, self._beta)

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._frequencies,1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._time_steps,1)))
            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H


class Semisupervised_NMF(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=0):
        super(Semisupervised_NMF, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):

        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W,self._H)
            self._H = update_H(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            self._W = update_W(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            cost = cost_fn(self._V, V_hat, self._beta)

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0],1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._H.shape[1],1)))
            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H

class Supervised_NMF_v1(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=0):
        super(Supervised_NMF_v1, self).__init__(W, H, X, epochs,
                     debug, beta=0)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W,self._H)
            self._H = update_H(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            self._W[:,1] = update_W(self._W, self._H,
                                       self._V, V_hat, self._beta)[:,1]

            V_hat = np.dot(self._W,self._H)
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0],1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._H.shape[1],1)))
            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

class Supervised_NMF_v2(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=0):
        super(Supervised_NMF_v2, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._H.shape[0]):
                self._H[i,:,:] = update_H(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(1, self._W.shape[0]):
                self._W[i,:,:] = update_W(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(np.sum(self._W,axis=0), axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0], self._W.shape[1], 1))
            self._H = self._H * np.tile(np.power(scale,-1),(self._H.shape[0]*self._H.shape[2],1)).reshape(self._H.shape)

            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

    def reconstruct(self,k, X):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
        W = self._W[k,:,:].reshape(self._W.shape[1], self._W.shape[2])
        H = self._H[k,:,:].reshape(self._H.shape[1], self._H.shape[2])
        return np.multiply((np.dot(W,H)/V_hat), X)

class Supervised_NMF_v3(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=0):
        super(Supervised_NMF_v3, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._H.shape[0]):
                self._H[i,:,:] = update_H(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(np.sum(self._W,axis=0), axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0], self._W.shape[1], 1))
            self._H = self._H * np.tile(np.power(scale,-1),(self._H.shape[0]*self._H.shape[2],1)).reshape(self._H.shape)

            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

    def reconstruct(self,k, X):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
        W = self._W[k,:,:].reshape(self._W.shape[1], self._W.shape[2])
        H = self._H[k,:,:].reshape(self._H.shape[1], self._H.shape[2])
        return np.multiply((np.dot(W,H)/V_hat), X)

class Supervised_NMF_v4(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=0):
        super(Supervised_NMF_v4, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._H.shape[0]):
                self._H[i,:,:] = update_H(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._W.shape[0]):
                self._W[i,:,:] = update_W(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(np.sum(self._W,axis=0), axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0], self._W.shape[1], 1))
            self._H = self._H * np.tile(np.power(scale,-1),(self._H.shape[0]*self._H.shape[2],1)).reshape(self._H.shape)

            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

    def reconstruct(self,k, X):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
        W = self._W[k,:,:].reshape(self._W.shape[1], self._W.shape[2])
        H = self._H[k,:,:].reshape(self._H.shape[1], self._H.shape[2])
        return np.multiply((np.dot(W,H)/V_hat), X)


class Semisupervised_CA_NMF(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=99):
        super(Semisupervised_CA_NMF, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):

        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W,self._H)
            self._H = update_H_CA(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            self._W = update_W_CA(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            cost = cost_fn(self._V, V_hat, self._beta)

            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0],1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._H.shape[1],1)))
            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H

class Supervised_CA_NMF_v1(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=99):
        super(Supervised_CA_NMF_v1, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(self._W,self._H)
            self._H = update_H_CA(self._W, self._H, self._V, V_hat, self._beta)

            V_hat = np.dot(self._W,self._H)
            self._W[:,1] = update_W_CA(self._W, self._H,
                                       self._V, V_hat, self._beta)[:,1]

            V_hat = np.dot(self._W,self._H)
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(self._W, axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0],1))
            self._H = self._H * np.transpose(np.tile(np.power(scale,-1),(self._H.shape[1],1)))
            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

class Supervised_CA_NMF_v2(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=99):
        super(Supervised_CA_NMF_v2, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._H.shape[0]):
                self._H[i,:,:] = update_H_CA(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(1, self._W.shape[0]):
                self._W[i,:,:] = update_W_CA(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(np.sum(self._W,axis=0), axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0], self._W.shape[1], 1))
            self._H = self._H * np.tile(np.power(scale,-1),(self._H.shape[0]*self._H.shape[2],1)).reshape(self._H.shape)

            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

    def reconstruct(self,k, X):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
        W = self._W[k,:,:].reshape(self._W.shape[1], self._W.shape[2])
        H = self._H[k,:,:].reshape(self._H.shape[1], self._H.shape[2])
        return np.multiply((np.dot(W,H)/V_hat), X)

class Supervised_CA_NMF_v3(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=99):
        super(Supervised_CA_NMF_v3, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._H.shape[0]):
                self._H[i,:,:] = update_H_CA(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))

            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(np.sum(self._W,axis=0), axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0], self._W.shape[1], 1))
            self._H = self._H * np.tile(np.power(scale,-1),(self._H.shape[0]*self._H.shape[2],1)).reshape(self._H.shape)

            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

    def reconstruct(self,k, X):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
        W = self._W[k,:,:].reshape(self._W.shape[1], self._W.shape[2])
        H = self._H[k,:,:].reshape(self._H.shape[1], self._H.shape[2])
        return np.multiply((np.dot(W,H)/V_hat), X)

class Supervised_CA_NMF_v4(beta_NMF):
    """docstring for beta_NMF"""
    def __init__(self, W, H, X, epochs=1000,
                 debug=False, beta=99):
        super(Supervised_CA_NMF_v4, self).__init__(W, H, X, epochs,
                     debug, beta)

    def train(self):
        loss = []
        for epoch in range(self._epochs):
            tick = time.time()

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._H.shape[0]):
                self._H[i,:,:] = update_H_CA(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            for i in range(self._W.shape[0]):
                self._W[i,:,:] = update_W_CA(self._W[i,:,:], self._H[i,:,:],
                                             self._V, V_hat, self._beta)

            V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
            cost = cost_fn(self._V, V_hat, self._beta)
            loss.append(cost)
            # print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))
            scale = np.sum(np.sum(self._W,axis=0), axis=0)
            self._W = self._W * np.tile(np.power(scale,-1),(self._W.shape[0], self._W.shape[1], 1))
            self._H = self._H * np.tile(np.power(scale,-1),(self._H.shape[0]*self._H.shape[2],1)).reshape(self._H.shape)

            if self._debug:
                print ('betaNMF -> iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost))


        return self._W, self._H, loss

    def reconstruct(self,k, X):
        # V_hat = th.dot(W, H)
        # W = W[:,k].reshape((-1,1))
        # H = H[k,:].reshape((1,-1))
        # return T.mul((T.dot(W,H)/V_hat),V)
        V_hat = np.dot(np.sum(self._W, axis=0),np.sum(self._H, axis=0))
        W = self._W[k,:,:].reshape(self._W.shape[1], self._W.shape[2])
        H = self._H[k,:,:].reshape(self._H.shape[1], self._H.shape[2])
        return np.multiply((np.dot(W,H)/V_hat), X)
