from run import do_experiment, get_features
from nmf import experiment_files_voc

from librosa import load, stft, istft, cqt,note_to_hz
from librosa.output import write_wav
from librosa.filters import window_bandwidth

import numpy as np

f1, f2 = experiment_files_voc()

x1, sr = load(f1, sr=16000)
x2, sr = load(f2, sr=16000)



print sr

nsamp = max(len(x1), len(x2))+1

x1 = np.pad(x1, (0, nsamp-len(x1)), mode='constant', constant_values=0)
x2 = np.pad(x2, (0, nsamp-len(x2)), mode='constant', constant_values=0)

x = x1 + x2

over_sample = 3
res_factor = 1

n_bins = int(8 * 11 * over_sample)
bins_per_octave=int(11*over_sample)


Q = float(res_factor) / (2.0**(1. / bins_per_octave) - 1)

freq = note_to_hz('C1') * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))
bit_for_check = (1 + 0.5 * window_bandwidth('hann') / Q)
print freq[-1] * bit_for_check
print freq[-1] * bit_for_check > 16000/2

X = cqt(x,
          sr=sr,
          hop_length=128,
          bins_per_octave=int(12*over_sample),
          n_bins=int(8 * 11 * over_sample),
          real=False,
          filter_scale=res_factor, 
          fmin=note_to_hz('C1'),
          scale=True)

X, nsamp, sr, x_stacked = get_features(True)

# SDR_w_is, SIR_w_is, SAR_w_is = do_experiment(cost_is, update_w_beta, update_h_beta, do_q_transform=True)