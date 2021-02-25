import numpy as np
import scipy.linalg as linalg
from sklearn.cluster import KMeans

from utils import stft, istft, write_wav, read_wav, determine_W
import matplotlib.pyplot as plt

import mir_eval
import cPickle

def NMF(file_1, file_2):
    O = 4275
    K = 2
    I = 513
    L = 5462
    WINDOW_SIZE = 1024
    fs = 16000
    iterations = 200
    # iterations = 1000
    # iterations = 1
    # iterations = 5


    max_theata=56
    max_azimuth=74


    # print "calculating W"
    W_ION = determine_W(I, O, fs, WINDOW_SIZE)

    # print "doing the STFT"
    x = file_1 + file_2
    X = stft(x)
    original_L = len(x)
    # print X.shape
    I, L = X.shape

    # print "starting the NMF"

    t = np.abs(np.random.rand(I, K)) + np.ones((I, K))
    v = np.abs(np.random.rand(K, L)) + np.ones((K, L))
    z = np.abs(np.random.rand(K, O)) + np.ones((K, O))
    W = np.abs(np.random.rand(I, O)) + np.ones((I, O))
    W = W_ION[:,:,0]

    W = W / np.linalg.norm(W)
    for i in range(iterations):
        # print i
        X_hat = sum(sum(z)) * np.sum(t,axis=1).reshape(I,1) * np.sum(v, axis=0).reshape(1,L)

        E = np.sum(W,axis=1).reshape(I,1) * np.sum(np.sum(z)) * np.sum(t, axis=1).reshape(I,1) * np.sum(v,axis=0).reshape(1,L)
        E = X - E
        # print sum(sum(E))
        tr = np.trace(np.dot(E.T, W))

        t = t * (1 + (np.sum(z, axis=1) * np.sum(v, axis=1) * tr) / (np.sum(z, axis=1) * np.sum(v, axis=1) * np.sum(X_hat, axis=1).reshape(I,1)))
        X_hat = sum(sum(z)) * np.sum(t,axis=1).reshape(I,1) * np.sum(v, axis=0).reshape(1,L)

        v = v * (1 + (np.sum(z, axis=1) * np.sum(t, axis=0) * tr).reshape(K,1) / (np.sum(z, axis=1) * np.sum(t, axis=0).reshape(1,K) * np.sum(X_hat, axis=0).reshape(L,1)).T)
        a_hat = np.sqrt(np.sum(v**2,axis=1)).reshape(K,1)
        v = v / a_hat
        t = t * a_hat.T
        X_hat = sum(sum(z)) * np.sum(t,axis=1).reshape(I,1) * np.sum(v, axis=0).reshape(1,L)

        z = z * (1 + (np.sum(t, axis=0) * np.sum(v, axis=1) * tr) / (np.sum(t, axis=0) * np.sum(v, axis=1) * np.sum(np.sum(X_hat)))).reshape(K,1)
        b_hat = np.sqrt(np.sum(z**2,axis=1)).reshape(K,1)
        z = z / b_hat
        t = t * b_hat.T
        X_hat = sum(sum(z)) * np.sum(t,axis=1).reshape(I,1) * np.sum(v, axis=0).reshape(1,L)

        A = (np.sum(z, axis=0).reshape(1,O) * np.sum(t, axis=1).reshape(I,1) * np.sum(np.sum(v)) * np.sum(X_hat, axis=1).reshape(I,1))
        B = (np.sum(z, axis=0).reshape(1,O) * np.sum(t, axis=1).reshape(I,1) * np.sum(np.sum(v)) * np.sum(E, axis=1).reshape(I,1))

        W_hat = W * (A + B)

        LV, D, RV = linalg.svd(W_hat)
        D_hat = np.zeros((I, O), dtype=complex)
        D_hat[:I,:I]=np.diag(D)
        D_hat = np.asarray(list(abs(n) for n in A))
        W_hat = np.dot(D_hat, RV)
        W_hat = np.dot(LV, W_hat)
        # print W_hat.shape

        W =  np.exp(1j * np.angle(W)) * np.absolute(W_hat)

        W = W / np.linalg.norm(W)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(z)
    prediction = kmeans.predict(z)
    b = np.diag([1, 1])
    B = sum(sum(b)) * sum(sum(z)) * np.sum(t, axis=1).reshape(I,1) * np.sum(v, axis=0).reshape(1,L)
    # for k in range(K):


    #     b_z = np.sum(b,axis=1) * sum(z[k])

    #     b_z_t0 = b_z[k] * np.sum(t, axis=1).reshape(I,1)

    #     b_z_t_v0 = b_z_t0 * np.sum(v, axis=0).reshape(1,L)

    #     A = b_z_t_v0

    #     C = A / B
    #     D = X * C
    #     print X.shape
    #     print C.shape




    #     result = istft(D, original_L)
    # #   print result.shape
    #     write_wav(save_name + '_mic_18_{}.wav'.format(k), result, fs)

    src_1 = X * (sum(z[0]) * t[:,0].reshape(I,1) * v[0].reshape(1,L) / B)
    src_2 = X * (sum(z[1]) * t[:,1].reshape(I,1) * v[1].reshape(1,L) / B)
    # '/Users/visionlab/Documents/MATLAB/Teun/source_seperation/NMF/IS-recordings/E_T2-win5760-is'
    ref_1 = file_1
    ref_2 = file_2
    bss = mir_eval.separation.bss_eval_sources(
                np.vstack((ref_1, ref_2)), np.vstack((istft(src_1, original_L), istft(src_2, original_L))))
    print 'SDR = {}, SIR = {}, SAR = {}, perm = {}'.format(*bss)


    # write_wav(save_name + '_mic_18_1_v2.wav', istft(src_1, original_L), fs)
    # write_wav(save_name + '_mic_18_2_v2.wav', istft(src_2, original_L), fs)
    return bss

def build_mix(design):
    print design
    train1, test1, train2, test2 = map(read_wav, design)
    nsamp = max(len(test1), len(test2))+1
    # time['mix'] = nsamp / float(voc_corpus.SAMPLE_RATE)
    # Simulate simple mic array with geometry with mic locations specified by the rows of constants.MIC_LOCS, sources
    # located in the far field in the -y and -x directions, respectively
    test1_ref = np.pad(test1, (0, nsamp-len(test1)), mode='constant', constant_values=0)
    test2_ref = np.pad(test2, (0, nsamp-len(test2)), mode='constant', constant_values=0)
    test1_delay = np.pad(test1, (1, nsamp-len(test1)-1), mode='constant', constant_values=0)
    test2_delay = np.pad(test2, (1, nsamp-len(test2)-1), mode='constant', constant_values=0)
    mix = np.vstack((test1_ref+test2_ref, test1_ref+test2_delay, test1_delay+test2_ref))

    return mix, test1_ref, test2_ref

def experiment_files_voc():
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    base = '/Volumes/Teun/vocalizationcorpus/data/'
    female = np.concatenate((np.arange(1,54), np.arange(72,202), np.arange(297,382), np.arange(460,554), np.arange(583,615), np.arange(655,709), np.arange(716,748), np.arange(816,900), np.arange(967,988), np.arange(1027,1067), np.arange(1225,1274), np.arange(1316,1344), np.arange(1368,1393), np.arange(1448,1466), np.arange(1472,1503), np.arange(1517,1525), np.arange(1532,1613), np.arange(1639,1689), np.arange(1712,1819), np.arange(2045,2060), np.arange(2095,2106), np.arange(2185,2225), np.arange(2250,2260), np.arange(2329,2453), np.arange(2545,2574), np.arange(2666,2676), np.arange(2717,2763)))
    male = np.concatenate((np.arange(54,72), np.arange(202,297), np.arange(382,460), np.arange(554,583), np.arange(615,655), np.arange(709,716), np.arange(748,816), np.arange(900,967), np.arange(988,1027), np.arange(1067, 1225), np.arange(1274, 1316), np.arange(1344, 1368), np.arange(1393, 1448), np.arange(1466, 1472), np.arange(1503, 1517), np.arange(1525, 1532), np.arange(1613, 1639), np.arange(1689, 1712), np.arange(1819, 2045), np.arange(2060, 2095), np.arange(2106, 2185), np.arange(2225, 2250), np.arange(2260, 2329), np.arange(2453, 2545), np.arange(2574, 2666), np.arange(2676, 2717)))

    male_file = np.random.choice(male, 2)
    female_file = np.random.choice(female, 2)

    return base + create_filename(male_file[0]), base + create_filename(male_file[1]), base + create_filename(female_file[0]), base + create_filename(female_file[1])

def experiment_files_bird():
    base = '/Volumes/Teun/1_WAV/'
    onlyfiles = [f for f in listdir(base) if isfile(join(base, f))]
    file_1 = numpy.random.choice(onlyfiles, 2)
    file_2 = numpy.random.choice(onlyfiles, 2)

    # while file_2[0].startswith(file_1[0][:-10]):
    #     file_1[0] = numpy.random.choice(onlyfiles, 1)[0]

    # while file_2[1].startswith(file_1[1][:-10]):
    #     file_1[1] = numpy.random.choice(onlyfiles, 1)[0]

    return base + file_1[0], base + file_1[1], base + file_2[0], base + file_2[1]

def create_filename(number):
    if number >= 1000:
        return 'S' + str(number) + '.wav'
    elif number >= 100:
        return 'S0' + str(number) + '.wav'
    elif number >= 10:
        return 'S00' + str(number) + '.wav'
    else:
        return 'S000' + str(number) + '.wav'

if __name__ == '__main__':
    n = 1000
    summary = {'sdr': np.zeros((n, 2)), 'sir': np.zeros((n, 2)),
                    'sar': np.zeros((n, 2)), 'perm': np.zeros((n, 2))}
    for i in range(n):
        design = experiment_files_voc()
        mix, test1_ref, test2_ref = build_mix(design)
        bss = NMF( test1_ref, test2_ref)

        summary['sdr'][i, :], summary['sir'][i, :],\
                    summary['sar'][i, :], summary['perm'][i, :] = bss


    f = open('summary_TDOA_voc.p', 'wb')
    cPickle.dump(summary, f)
    f.close()
