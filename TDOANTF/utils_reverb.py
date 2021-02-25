from librosa import load, stft, istft, amplitude_to_db, db_to_amplitude
from librosa.display import specshow
from librosa.output import write_wav
import pyroomacoustics as pra
import numpy as np

def load_files(files):
    s1, _  = load(files[0], sr=16000)
    s2, _  = load(files[1], sr=16000)

    if len(s1) > len(s2):
        pad_length = len(s1) - len(s2)
        s2 = np.pad(s2, (0, pad_length), 'reflect')
    elif len(s2) > len(s1):
        pad_length = len(s2) - len(s1)
        s1 = np.pad(s1, (0, pad_length), 'reflect')

    return s1, s2

def do_reverb(s1, s2, locs=[[8.,4.,1.6],[4.,6.,1.6]]):
    corners = np.array([[0,0], [0,8], [8,8], [8,0]]).T  # [x,y]
    room = pra.Room.from_corners(corners)
    room.extrude(5.)

    room.add_source(locs[0], signal=s1)
    room.add_source(locs[1], signal=s2)
    #[[X],[Y],[Z]]
    R = np.asarray([[4.75,5.5],[2.,2.],[1.,1]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    room.simulate()

    return room, locs

def do_stft(s1, s2, room):
    nfft=1024
    win = 256
    hop = 128

    Y1 = stft(room.mic_array.signals[0,:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    Y2 = stft(room.mic_array.signals[1,:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    X1 = stft(s1, n_fft=nfft, hop_length=hop, win_length=win)
    X2 = stft(s2, n_fft=nfft, hop_length=hop, win_length=win)

    return Y1, Y2, X1, X2

def do_stft_2mic(m1, m2, s1, s2):
    nfft=1024
    win = 256
    hop = 128
    shortest = len(s1)
    if len(s2) < shortest:
        shortest = len(s2)
    Y1 = stft(m1[:shortest], n_fft=nfft, hop_length=hop, win_length=win)
    Y2 = stft(m2[:shortest], n_fft=nfft, hop_length=hop, win_length=win)
    X1 = stft(s1[:shortest], n_fft=nfft, hop_length=hop, win_length=win)
    X2 = stft(s2[:shortest], n_fft=nfft, hop_length=hop, win_length=win)

    return Y1, Y2, X1, X2

def signum(X):
    absX = np.abs(X)
    return np.multiply(np.sqrt(absX), np.divide(X, absX))
