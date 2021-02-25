import time
import numpy as np
import theano as th
from theano import tensor as T

from librosa import load, stft, istft, cqt,note_to_hz
from librosa.output import write_wav

from icqt import icqt_recursive as icqt

th.config.optimizer='None'
th.config.exception_verbosity='high'

from update_rules import update_h_beta, update_h_cauchy, update_w_beta, update_w_cauchy
from cost import cost_cau, cost_is
from nmf import NMF, experiment_files_voc
from Deep_NMF import Deep_NMF
from Deep_NMF_GD import Deep_NMF_GD

import mir_eval

if __name__ == '__main__':
    files = 100
    SDR_1 = 0
    SIR_1 = 0
    SAR_1 = 0

    SDR_2 = 0
    SIR_2 = 0
    SAR_2 = 0

    SDR_3 = 0
    SIR_3 = 0
    SAR_3 = 0

    SDR_4 = 0
    SIR_4 = 0
    SAR_4 = 0

    SDR_5 = 0
    SIR_5 = 0
    SAR_5 = 0

    SDR_6 = 0
    SIR_6 = 0
    SAR_6 = 0

    SDR_7 = 0
    SIR_7 = 0
    SAR_7 = 0

    SDR_8 = 0
    SIR_8 = 0
    SAR_8 = 0

    # x, sr = load('Hannah_teun_317Mic18.WAV')
    for file_index in range(files):
      f1, f2 = experiment_files_voc()
      x1, sr = load(f1)
      x2, sr = load(f2)

      nsamp = max(len(x1), len(x2))+1

      x1 = np.pad(x1, (0, nsamp-len(x1)), mode='constant', constant_values=0)
      x2 = np.pad(x2, (0, nsamp-len(x2)), mode='constant', constant_values=0)

      x = x1 + x2

      X = stft(x, win_length=256,hop_length=128, n_fft=1024)
      
      V = np.abs(X)**2
      
      frequencies, time_steps = X.shape
      sources = 2
      print "beta NMF"
      nmf = NMF(frequencies, time_steps, sources, V)
      train = nmf.train(cost_is, update_w_beta, update_h_beta)

      new_x1 = np.real(istft(nmf.reconstruct_func(0, X), win_length=256,hop_length=128))
      new_x2 = np.real(istft(nmf.reconstruct_func(1, X), win_length=256,hop_length=128))


      if len(new_x1) > nsamp:
            new_x1 = new_x1[:nsamp,:]
            new_x2 = new_x2[:nsamp,:]
      else:
            
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      print len(new_x1)
      print len(new_x2)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_1 += bss[0]
      SIR_1 += bss[1]
      SAR_1 += bss[2]
      print file_index
      print "cauchy NMF"
      nmf = NMF(frequencies, time_steps, sources, V)
      train = nmf.train(cost_cau, update_w_cauchy, update_h_cauchy)

      new_x1 = np.real(istft(nmf.reconstruct_func(0, X), win_length=256,hop_length=128))
      new_x2 = np.real(istft(nmf.reconstruct_func(1, X), win_length=256,hop_length=128))

      if len(new_x1) > len(x1):
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_nsamp = max(len(new_x1), len(new_x2))+1
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_2 += bss[0]
      SIR_2 += bss[1]
      SAR_2 += bss[2]
      
      print "Deep cauchy nmf"
      nmf = Deep_NMF(frequencies, time_steps, sources, 10, V)
      train = nmf.train(cost_cau, update_w_cauchy, update_h_cauchy)

      new_x1 = np.real(istft(nmf.reconstruct_func(0, X), win_length=256,hop_length=128))
      new_x2 = np.real(istft(nmf.reconstruct_func(1, X), win_length=256,hop_length=128))

      if len(new_x1) > nsamp:
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_6 += bss[0]
      SIR_6 += bss[1]
      SAR_6 += bss[2]
      print "Deep beta NMF"
      nmf = Deep_NMF(frequencies, time_steps, sources, 10, V)
      train = nmf.train(cost_is, update_w_beta, update_h_beta)

      new_x1 = np.real(istft(nmf.reconstruct_func(0, X), win_length=256,hop_length=128))
      new_x2 = np.real(istft(nmf.reconstruct_func(1, X), win_length=256,hop_length=128))

      if len(new_x1) > nsamp:
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_7 += bss[0]
      SIR_7 += bss[1]
      SAR_7 += bss[2]
      

      over_sample = 3
      res_factor = 1
      X = cqt(x,
                  sr=sr,
                  hop_length=128,
                  bins_per_octave=int(12*over_sample),
                  n_bins=int(8 * 12 * over_sample),
                  real=False,
                  filter_scale=res_factor, 
                  fmin=note_to_hz('C1'),
                  scale=True)
      V = np.abs(X)

      frequencies, time_steps = X.shape
      sources = 2

      print "beta wavelet nmf"
      nmf = NMF(frequencies, time_steps, sources, V)
      train = nmf.train(cost_is, update_w_beta, update_h_beta)

      new_x1 = new_x = icqt(nmf.reconstruct_func(0, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)
      new_x2 = new_x = icqt(nmf.reconstruct_func(1, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)


      if len(new_x1) > len(x1):
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_nsamp = max(len(new_x1), len(new_x2))+1
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_3 += bss[0]
      SIR_3 += bss[1]
      SAR_3 += bss[2]

      print "cauchy wavelet NMF"
      nmf = NMF(frequencies, time_steps, sources, V)
      train = nmf.train(cost_cau, update_w_cauchy, update_h_cauchy)

      new_x1 = new_x = icqt(nmf.reconstruct_func(0, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)
      new_x2 = new_x = icqt(nmf.reconstruct_func(1, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)

      if len(new_x1) > nsamp:
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_4 += bss[0]
      SIR_4 += bss[1]
      SAR_4 += bss[2]

      print " wavelet Deep cauchy NMF"

      nmf = Deep_NMF(frequencies, time_steps, sources, 10, V, epochs=32)
      train = nmf.train(cost_cau, update_w_cauchy, update_h_cauchy)

      new_x1 = new_x = icqt(nmf.reconstruct_func(0, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)
      new_x2 = new_x = icqt(nmf.reconstruct_func(1, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)

      if len(new_x1) > nsamp:
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_5 += bss[0]
      SIR_5 += bss[1]
      SAR_5 += bss[2]

      print "wavelet Deep beta nmf"

      nmf = Deep_NMF(frequencies, time_steps, sources, 10, V, epochs=50)
      train = nmf.train(cost_is, update_w_beta, update_h_beta)

      new_x1 = new_x = icqt(nmf.reconstruct_func(0, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)
      new_x2 = new_x = icqt(nmf.reconstruct_func(1, X), sr=sr,
                    hop_length=128,
                    bins_per_octave=int(12 * over_sample),
                    filter_scale=res_factor,
                    fmin=note_to_hz('C1'),
                   scale=True)

      if len(new_x1) > nsamp:
            new_x1 = new_x1[:nsamp]
            new_x2 = new_x2[:nsamp]
      else:
            new_x1 = np.pad(new_x1, (0, nsamp-len(new_x1)), mode='constant', constant_values=0)
            new_x2 = np.pad(new_x2, (0, nsamp-len(new_x2)), mode='constant', constant_values=0)

      bss = mir_eval.separation.bss_eval_sources(
                  np.vstack((x1, x2)), np.vstack((new_x1, new_x2)))
      SDR_8 += bss[0]
      SIR_8 += bss[1]
      SAR_8 += bss[2]

    print "beta stft SDR: " + str(SDR_1 / files)
    print "beta stft SIR: " + str(SIR_1 / files)
    print "beta stft SAR: " + str(SAR_1 / files) 

    print "cauchy stft SDR: " + str(SDR_2 / files)
    print "cauchy stft SIR: " + str(SIR_2 / files)
    print "cauchy stft SAR: " + str(SAR_2 / files) 

    print "beta wavelet SDR: " + str(SDR_3 / files)
    print "beta wavelet SIR: " + str(SIR_3 / files)
    print "beta wavelet SAR: " + str(SAR_3 / files) 

    print "cauchy wavelet SDR: " + str(SDR_4 / files)
    print "cauchy wavelet SIR: " + str(SIR_4 / files)
    print "cauchy wavelet SAR: " + str(SAR_4 / files) 

    print "cauchy stft SDR: " + str(SDR_6 / files)
    print "cauchy stft SIR: " + str(SIR_6 / files)
    print "cauchy stft SAR: " + str(SAR_6 / files) 

    print "beta stft SDR: " + str(SDR_7 / files)
    print "beta stft SIR: " + str(SIR_7 / files)
    print "beta stft SAR: " + str(SAR_7 / files) 

    print "cauchy wavelet SDR: " + str(SDR_5 / files)
    print "cauchy wavelet SIR: " + str(SIR_5 / files)
    print "cauchy wavelet SAR: " + str(SAR_5 / files) 

    print "beta wavelet SDR: " + str(SDR_8 / files)
    print "beta wavelet SIR: " + str(SIR_8 / files)
    print "beta wavelet SAR: " + str(SAR_8 / files) 