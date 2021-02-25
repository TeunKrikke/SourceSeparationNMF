import time


import numpy as np

import tensorflow as tf

from librosa import load, stft, istft
from librosa.output import write_wav

class SummaryTest(object):
    """docstring for SummaryTest"""
    def __init__(self, frequencies, time_steps):
        super(SummaryTest, self).__init__()
        self._frequencies = frequencies
        self._time_steps = time_steps
        self.epochs=100
        
        
    
    def train(self, X):
        with tf.Graph().as_default():
            session = tf.Session()
            V = tf.placeholder(tf.float32, shape=(self._frequencies, self._time_steps), name='V')

            with tf.name_scope('temp') as scope:
                V = V*0.5
                tf.summary.audio('V', V, 22050, max_outputs=1)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./train',
                                      session.graph)


            init_op = tf.global_variables_initializer()
            session.run(init_op)
            for epoch in range(self.epochs):
                summary, _ = session.run([merged, V], feed_dict={V:X})
                train_writer.add_summary(summary, epoch)

if __name__ == '__main__':
    x, sr = load('data.wav')
    print sr
    X = stft(x, win_length=256,hop_length=128, n_fft=1024)
    X= np.abs(X)**2
    
    frequencies, time_steps = X.shape
    test = SummaryTest(frequencies, time_steps)

    test.train(X)

