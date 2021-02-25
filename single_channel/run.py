from update_rules import update_h_beta, update_h_cauchy, update_w_beta
from update_rules import update_w_cauchy
from cost import cost_cau, cost_is, cost_kl, cost_euc
from nmf import NMF, experiment_files_voc
from Deep_NMF import Deep_NMF
from utils import write_wavefile

import numpy as np
import theano as th

from librosa import load, stft, istft

import mir_eval

th.config.optimizer = 'None'
th.config.exception_verbosity = 'high'


def get_features(do_q_transform=False):
    """
    This function does the transform to create frequency domain data

    Keyword arguments:
    do_q_transform -- do the Q-transform incase we are working with wavelets.
    """

    f1, f2 = experiment_files_voc()
    # get a couple of filenames that we want to load
    # loading the file. The sr needs to be given otherwise
    # it will use the standard sr of 22050
    x1, sr = load(f1, sr=16000)
    x2, sr = load(f2, sr=16000)

    # determine the length of the longest file
    # plus one to make it end in a zero
    nsamp = max(len(x1), len(x2)) + 1
    # zeropad the file to length the longest file
    x1 = np.pad(x1, (0, nsamp - len(x1)), mode='constant', constant_values=0)
    x2 = np.pad(x2, (0, nsamp - len(x2)), mode='constant', constant_values=0)

    x = x1 + x2  # make the mixture
    X = stft(x, win_length=256, hop_length=128, n_fft=1024)
    x_stacked = np.vstack((x1, x2))  # stack the originals
    return X, nsamp, sr, x_stacked


def go_back(X_s, nsamp, sr, do_q_transform=False):
    """
    This function does the inverse transform to create time domain data

    Keyword arguments:
    X_s -- the frequency domain data
    nsamp -- the number of samples of the original data to make sure we can
    compare the transformed data with the original data
    sr -- the sample rate needed for the inverse q-transform
    do_q_transform -- do the inverse Q-transform incase we are working with
    wavelets.
    """
    x_s = []  # transformed data per source
    for X_sa in X_s:  # for each transformed data set
        new_x = np.real(istft(X_sa, win_length=256, hop_length=128))
        if len(new_x) > nsamp:
            # when the inverse is longer than the original then cut it
            new_x = new_x[:nsamp]
        else:  # otherwise lengthen it to the size of the original
            new_x = np.pad(new_x, (0, nsamp - len(new_x)),
                           mode='constant', constant_values=0)

        x_s.append(new_x)  # add it to the list
    return np.stack(x_s)


def do_experiment(cost_func, update_w, update_h, do_q_transform=False,
                  go_deep=False, files=100, convolution=False, norm_W=0,
                  norm_H=0, beta=0):
    """run a single function over multiple files.
    This function runs an experiment which means that it runs one single NMF
    over multiple files

    Keyword arguments:
    cost_func -- the cost function to be tested
    update_w -- the W update rule that is associated with the cost function
    update_h -- the H update rule that is associated with the cost fucntion
    do_q_transform -- do the wavelet transform of the input data instead of
    the STFT (default False)
    go_deep -- make a multi layer NMF instead of a single layer (default False)
    files -- run the cost function over this number of files (default 100)
    convolution -- do convolution instead of vanilla NMF (default False)
    norm_W -- normalise the W matrix with L1 norm (1) or L2 norm (2)
    (default 0)
    norm_H -- normalise the H matrix with L1 norm (1) or L2 norm (2)
    (default 0)
    beta -- the beta parameter that determines which update rule to use
    Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
    """

    SDR = 0  # signal to distortion ratio over all files summed
    SAR = 0  # signal to artifact ratio over all files summed
    SIR = 0  # signal to inference ratio over all files summed

    for file_index in range(files):
        X, nsamp, sr, x_stacked = get_features(do_q_transform)
        # get the input features for the algorithm

        V = X.real**2
        # determine the magnitude of the input to make sure that everything is
        # non negative and remover the imaginary part

        frequencies, time_steps = X.shape
        # get the shape of the input which
        # we use to create our W and H matrices
        sources = 2  # we are working with 2 sources
        if go_deep:  # if we are working with a multilayer NMF then use this.
            # We train the deep NMF for 32 epochs to avoid overfitting
            nmf = Deep_NMF(frequencies, time_steps, sources, 10, V, epochs=32)
            nmf.train(cost_func, update_w, update_h, V)
        else:  # otherwise it is a single layer NMF
            nmf = NMF(frequencies, time_steps, sources, V, conv=convolution,
                      beta=beta)
            # train the NMF algorithm
            nmf.train(cost_func, update_w, update_h, beta=beta,
                      convolution=convolution, norm_W=norm_W,
                      norm_H=norm_W)
        X_s = []
        for s in range(sources):  # reconstruct the sources for evaluation
            X_s.append(nmf.reconstruct_func(s, X))

        x_s = go_back(X_s, nsamp, sr, do_q_transform)
        # get the inverse STFT or the inverse wavelet
        # transform from the separated data

        bss = mir_eval.separation.bss_eval_sources(x_stacked, x_s)
        # compare the separated (x_s) with the ground truth (x_stacked)
        # and get the improvement

        SDR += bss[0]
        SIR += bss[1]
        SAR += bss[2]

    return SDR, SIR, SAR


def do_experiment_batch(do_q_transform=False, convolution=False,
                        norm_W=0, norm_H=0, go_deep=False):
    """run a batch of experiments over multiple files.
    This function runs all cost functions over the data. It call do experiment
    with the specified parameters

    Keyword arguments:
    do_q_transform -- do the wavelet transform of the input data instead of the
    STFT (default False)
    go_deep -- make a multi layer NMF instead of a single layer (default False)
    convolution -- do convolution instead of vanilla NMF (default False)
    norm_W -- normalise the W matrix with L1 norm (1) or L2 norm (2)
    (default 0)
    norm_H -- normalise the H matrix with L1 norm (1) or L2 norm (2)
    (default 0)

    """

    files = 100  # number of files we want to use

    # print("Vanilla techniques IS, CAU, EUC, KL")
    # get all the cost functions and put them in order of the beta parameter
    cost_funcs = [cost_is, cost_kl, cost_euc, cost_cau]
    for beta, cost in enumerate(cost_funcs):
        print beta  # print the beta param so we know which cost function
        # we are looking at
        if beta < 3:  # everthing lower than 3 means that we use
            # the beta update
            SDR, SIR, SAR = do_experiment(cost_funcs[beta], update_w_beta,
                                          update_h_beta, do_q_transform,
                                          go_deep, files, convolution, norm_W,
                                          norm_H, beta)
        else:  # otherwise we are working with the cauchy update function
            SDR, SIR, SAR = do_experiment(cost_cau, update_w_cauchy,
                                          update_h_cauchy)
        print SDR / files, SIR / files, SAR / files


class Sanity_check(object):
    """
    do a sanity test on the algorithms by using the sample file given
    by the matlab IS algorithm
    """
    def __init__(self):
        super(Sanity_check, self).__init__()

    def get_features(self):
        """
        This function does the transform to create frequency domain data

        Keyword arguments:
        do_q_transform -- do the Q-transform incase we are working with
        wavelets.
        """

        # loading the file. The sr needs to be given otherwise it will use the
        # standard sr of 22050
        x1, sr = load('data.wav', sr=16000)

        # determine the length of the longest file plus one to make it
        # end in a zero
        nsamp = len(x1) + 1
        # zeropad the file to length the longest file
        x1 = np.pad(x1, (0, nsamp - len(x1)),
                    mode='constant', constant_values=0)

        x = x1

        X = stft(x, win_length=256, hop_length=128, n_fft=256)

        return X, nsamp, sr

    def do_experiment(self, cost_func, update_w, update_h,
                      do_q_transform=False,
                      go_deep=False, files=1, convolution=False, norm_W=0,
                      norm_H=0, beta=0):
        """run a single function over multiple files.
        This function runs an experiment which means that it runs one single
        NMF over multiple files

        Keyword arguments:
        cost_func -- the cost function to be tested
        update_w -- the W update rule that is associated with the cost function
        update_h -- the H update rule that is associated with the cost fucntion
        do_q_transform -- do the wavelet transform of the input data instead
        of the STFT (default False)
        go_deep -- make a multi layer NMF instead of a single layer
        (default False)
        files -- run the cost function over this number of files (default 100)
        convolution -- do convolution instead of vanilla NMF (default False)
        norm_W -- normalise the W matrix with L1 norm (1) or L2 norm (2)
        (default 0)
        norm_H -- normalise the H matrix with L1 norm (1) or L2 norm (2)
        (default 0)
        beta -- the beta parameter that determines which update rule to use
        Eucledian (2) Kullback-Leibler (1) Itakura-Saito (0) (default 0)
        """

        for file_index in range(files):
            X, nsamp, sr = self.get_features()
            # get the input features for the algorithm

            V = X.real**2
            # determine the magnitude of the input to make sure that
            # everything is non negative and remover the imaginary part

            frequencies, time_steps = X.shape
            # get the shape of the input which we use to create our
            # W and H matrices
            sources = 10  # we are working with 10 sources
            if go_deep:  # if we are working with a multilayer NMF then use
                # this. We train the deep NMF for 32 epochs to avoid
                # overfitting
                nmf = Deep_NMF(frequencies, time_steps, sources, 10, V,
                               epochs=32)
            else:  # otherwise it is a single layer NMF
                nmf = NMF(frequencies, time_steps, sources, V,
                          conv=convolution, beta=beta)
            # train the NMF algorithm
            nmf.train(cost_func, update_w, update_h, beta=beta,
                      convolution=convolution, norm_W=norm_W, norm_H=norm_W)
            X_s = []
            for s in range(sources):  # reconstruct the sources for evaluation
                X_s.append(nmf.reconstruct_func(s, X))

            x_s = go_back(X_s, nsamp, sr, do_q_transform)
            # get the inverse STFT or the inverse wavelet transform
            # from the separated data
            write_wavefile(x_s)


if __name__ == '__main__':
    print "Vanilla STFT"
    do_experiment_batch(go_deep=True)
    # # print "Vanilla Q_transform"
    # # do_experiment_batch(do_q_transform=True)
    # print "Conv STFT"
    # do_experiment_batch(do_q_transform=False, convolution=True)

    # check = Sanity_check()
    # check.do_experiment(cost_is, update_w_beta, update_h_beta, beta=0)
    # check.do_experiment(cost_cau, update_w_cauchy, update_h_cauchy, beta=3)
