import numpy as np
import scipy
import scipy.io.wavfile
import time
from librosa import load

import glob
import os

# from timit import SAMPLE_RATE, BITS_PER_SAMPLE

EPSI = 1e-12 # constant to avoid division by zero
D = 48 # 24 # Number of quantization levels for DOA angle
WINDOW_SIZE = 1024 # window for the STFT
HOP_SIZE = 256 # overlap for the STFT
SAMPLE_RATE = 16000 # voc corpus
SPEED_OF_SOUND = 342.0
MIC_SEP = SPEED_OF_SOUND/SAMPLE_RATE  # Separation between mics in meters corresponding to a one-sample delay
MIC_LOCS = np.array([[0, 0, 0], [MIC_SEP, 0, 0], [0, MIC_SEP, 0]])

__author__ = 'tfk, nstein'

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def determine_pinv(mics):
    locs = []
    for mic in mics:
        locs.append(mic.position)

    locs = np.asarray(locs)

    pinv = np.linalg.pinv(locs[1:, :2] - locs[0, :2])

    return pinv

def normalize(x, axis=None):
    return x / (np.sum(x, axis, keepdims=True) + EPSI)



def stft(x, window=None):
    if window is None:
        window = np.hamming(WINDOW_SIZE)
    x_stft = np.array(
        [scipy.fft(window * x[i:i + WINDOW_SIZE])[:WINDOW_SIZE / 2 + 1] for i in xrange(0, len(x) - WINDOW_SIZE, HOP_SIZE)])
    return x_stft.T


def istft(x_stft, ts, window=None):
    x = np.zeros(ts)
    if window is None:
        window = np.hamming(WINDOW_SIZE)
    for n, i in enumerate(range(0, len(x) - WINDOW_SIZE, HOP_SIZE)):
        x[i:i + WINDOW_SIZE] += \
            window * np.real(scipy.ifft(np.concatenate((x_stft[:, n], np.flipud(x_stft[1:-1, n]).conj()))))
    return x


def apply_masks(stft_to_mask, masks, ts, K):
    return np.vstack([istft(mask[:, :, 0]*stft_to_mask, ts) for mask in np.split(masks, K, axis=2)])


def fast_doa(angles, pinv):
    angles_diff = (angles[1:, :] - angles[0, :] + np.pi) % (2 * np.pi) - np.pi
    alpha = np.dot(pinv, angles_diff)
    return np.arctan2(-alpha[1, :], -alpha[0, :])


def extract_stft_pobs_and_d(mix, mics, mic_nr=0):
    mic_count = len(mix)

    start = time.time()
    stfts = np.dstack([stft(x.flatten()) for x in np.split(mix, mic_count, axis=0)])

    pobs = normalize(np.abs(stfts[:, :, mic_nr]))

    after_stfts = time.time()
    F, T = pobs.shape
    angles = np.angle(stfts)
    doa = fast_doa(angles.reshape(F*T, mic_count).T, determine_pinv(mics)).reshape(F, T)
    d = np.floor(D*(doa + np.pi)/(2*np.pi+EPSI))
    after_doa = time.time()
    return stfts[:, :, mic_nr], pobs, d, after_stfts-start, after_doa-after_stfts

def multi_extract(signal, mics):
    mic_count = len(mix)

    start = time.time()
    stfts = np.dstack([stft(x.flatten()) for x in np.split(mix, mic_count, axis=0)])

    pobs = normalize(np.abs(stfts))


    after_stfts = time.time()
    F, T = pobs.shape
    angles = np.angle(stfts)
    doa = fast_doa(angles.reshape(F*T, mic_count).T, determine_pinv(mics)).reshape(F, T)
    d = np.floor(D*(doa + np.pi)/(2*np.pi+EPSI))
    after_doa = time.time()
    return stfts, pobs, d, after_stfts-start, after_doa-after_stfts



def read_wav(filename):
    # _, data = scipy.io.wavfile.read(filename)
    data, _ = load(filename)
    # return np.array(data, dtype='float')/(2**(BITS_PER_SAMPLE-1))
    return data


def write_wav(filename, data, sample_rate):
    scipy.io.wavfile.write(filename, sample_rate, 0.9*data/np.abs(data).max())


def determine_A(F, fs, N, max_theta=56,max_azimuth=66,radius=1):
    speed_of_sound = 344
    O = max_azimuth * max_theta
    A = np.zeros((F, O, np.power(len(MIC_LOCS),2)), dtype="complex")

    # for theta in range(max_theta):
    #     for azimuth in range(max_azimuth):
    #         k_o = np.asarray([np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(azimuth)),radius])
    #         for n in range(len(MIC_LOCS)):
    #             tav_n = np.negative(k_o).T.dot(MIC_LOCS[n]) / speed_of_sound
    #             for m in range(len(MIC_LOCS)):
    #                 tav_m = np.negative(k_o).T.dot(MIC_LOCS[m]) / speed_of_sound
    #                 for i in range(F):
    #                     f_i = (i - 1) * fs/F
    #                     W[i, (theta*max_theta) + azimuth, (n * len(MIC_LOCS))+m] = np.exp(1j*2*np.pi*f_i*tav_n - tav_m)

    azimuth = 0
    theta = 0
    for ta in range(max_theta * max_azimuth):
        if azimuth == max_azimuth:
            theta += 1
            azimuth = 0
        n = 0
        m = 0
        k_o = np.asarray([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(azimuth)), radius])
        for nm in range(len(MIC_LOCS) * len(MIC_LOCS)):
            if m == len(MIC_LOCS):
                n += 1
                m = 0
            tav_n = np.negative(k_o).T.dot(MIC_LOCS[n]) / speed_of_sound
            tav_m = np.negative(k_o).T.dot(MIC_LOCS[m]) / speed_of_sound
            for i in range(F):
                f_i = (i - 1) * fs / F
                A[i, ta, nm] = np.exp(1j * 2 * np.pi * f_i * tav_n - tav_m)
        azimuth += 1

    return A
