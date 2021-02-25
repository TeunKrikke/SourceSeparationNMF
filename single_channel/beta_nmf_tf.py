'''
beta_NMF.py
'''
import time
import numpy as np
import tensorflow as tf

from librosa import load, stft, istft
from librosa.output import write_wav

class beta_NMF(object):
    """docstring for beta_NMF"""
    def __init__(self, frequencies, time_steps, sources):
        super(beta_NMF, self).__init__()
        self._frequencies = frequencies
        self._time_steps = time_steps
        self._sources = sources
        self._epochs = 1000

        self.H = np.asarray(np.random.rand(self._sources, self._time_steps)+np.ones((self._sources, self._time_steps)), dtype=np.float32)
        self.W = np.asarray(np.random.rand(self._frequencies, self._sources)+np.ones((self._frequencies, self._sources)), dtype=np.float32)
        self.cost_v = []


    def train(self, _V, _X):
        g = tf.Graph()
        with g.as_default():
            V = tf.placeholder(tf.float32, shape=(self._frequencies, self._time_steps), name='V')
            X = tf.placeholder(tf.float32, shape=(self._frequencies, self._time_steps), name='X')
            W = tf.placeholder(tf.float32, shape=(self._frequencies, self._sources), name="W")
            H = tf.placeholder(tf.float32, shape=(self._sources, self._time_steps), name="H")

            V_hat = tf.matmul(W, H)
            with tf.name_scope('update_H') as scope:
                update_H = tf.multiply(H, tf.matmul(tf.transpose(W), tf.multiply(V, tf.pow(V_hat,-2))) / tf.matmul(tf.transpose(W), tf.pow(V_hat,-1)))
                tf.summary.histogram("H",H)

            V_hat = tf.matmul(W, H)
            with tf.name_scope('update_W') as scope:
                update_W = tf.multiply(W, tf.matmul(tf.multiply(V, tf.pow(V_hat,-2)), tf.transpose(H)) / tf.matmul(tf.pow(V_hat,-1), tf.transpose(H)))
                tf.summary.histogram("W",W)

            V_hat = tf.matmul(W, H)
            with tf.name_scope('cost') as scope:
                cost = tf.reduce_sum(V/V_hat - tf.log(V/V_hat)) - 1
                tf.summary.scalar("cost",cost)

            tf.summary.histogram("V_hat",V_hat)

            V_hat = tf.matmul(W, H)
            W_s = tf.reshape(W[:,0], [-1,1])
            H_s = tf.reshape(H[0,:],[1,-1])
            result = tf.multiply((tf.matmul(W_s,H_s)/V_hat),X)

            tf.summary.histogram("Source_0",result)
            tf.summary.audio('Audio_source_0', result, 22050, max_outputs=1)
        with tf.Session(graph=g) as self._session:
            merged = tf.summary.merge_all()

            train_writer = tf.summary.FileWriter('./train_beta_nmf',
                                      self._session.graph)

            init_op = tf.global_variables_initializer()
            self._session.run(init_op)
            self._epochs = 1000

            for epoch in range(self._epochs):
                tick = time.time()
                self.H = self._session.run([update_H], feed_dict={V:_V, H:self.H, W:self.W})[0]
                self.W = self._session.run([update_W], feed_dict={V:_V, H:self.H, W:self.W})[0]
                cost_value, summary = self._session.run([cost,merged], feed_dict={V:_V, X:_X, H:self.H, W:self.W})
                self.cost_v.append(cost_value)
                train_writer.add_summary(summary, epoch)
                print ('iter {} time it took {}ms. This resulted in a loss of {}'.format(epoch, (time.time() - tick) * 1000, cost_value))

    def reconstruct(self, X, k):
        V = tf.placeholder(tf.float32, shape=(self._frequencies, self._time_steps), name='V')
        V_hat = tf.matmul(self.W, self.H)
        W = tf.reshape(self.W[:,k], [-1,1])
        H = tf.reshape(self.H[k,:],[1,-1])
        result = tf.multiply((tf.matmul(W,H)/V_hat),V)
        return self._session.run([result], feed_dict={V:X})[0]

if __name__ == '__main__':
    # x, sr = load('data.wav')
    x, sr = load('/home/tinus/Workspace/tensorflow_ws/Source_separation/NMF/H_T_200Mic1NNE.wav')
    X = stft(x, win_length=256,hop_length=128, n_fft=1024)
    V = np.abs(X)**2

    frequencies, time_steps = X.shape
    sources = 2
    nmf = beta_NMF(frequencies, time_steps, sources)
    nmf.train(V,X)

    for s in range(sources):
        new_x = np.real(istft(nmf.reconstruct(X,s), win_length=256,hop_length=128))
        write_wav('sources/tf_separated_{}.wav'.format(s), new_x, sr)
