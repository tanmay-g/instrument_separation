import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.io.wavfile import read, write


def separate_instruments(file_name = "rhythm_birdland.wav"):

    # Read the file
    fs, x = read("./inputs/" + file_name)

    winlen = 1024

    # Step 1: Generate STFT

    # Have used terminology from the paper
    # h and i represent indices of frequency and time bins
    h, i, F = stft(x=x, fs=fs, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
                   nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)

    # Step 2: Calculate a range-compressed version of the power spectrogram
    gamma = 0.3
    W = np.power(np.abs(F), 2 * gamma)

    # Step 3: Initialise
    k_max = 50
    H = 0.5 * W
    P = 0.5 * W
    alpha = 0.3

    for k in range(k_max):
        # Step 4: Calculate update variable delta
        term_1 = np.zeros_like(H)
        term_2 = np.zeros_like(H)

        for i_iter in range(1, np.shape(H)[1]-1):
            term_1[:, i_iter] = alpha * ((H[:, i_iter-1] + H[:, i_iter+1] - (2 * H[:, i_iter])) / 4)
        term_1[:, 0] = alpha * ((H[:, 1] - H[:, 0]) / 2)
        term_1[:, -1] = alpha * ((H[:, -2] - H[:, -1]) / 2)

        for h_iter in range(1, np.shape(H)[0]-1):
            term_2[h_iter, :] = (1 - alpha) * ((P[h_iter-1, :] + P[h_iter+1, :] - (2 * P[h_iter, :])) / 4)
        term_2[0, :] = (1 - alpha) * ((P[1, :] - P[0, :]) / 2)
        term_2[-1, :] = (1 - alpha) * ((P[-2, :] - P[-1, :]) / 2)

        delta = term_1 - term_2
        # Reduce "step size"
        delta = delta * 0.9

        # Step 5: Update H and P
        H = np.minimum(np.maximum(H + delta, 0), W)
        P = W - H

        # Step 6: Increment k (automatically through loop)

    # Step 7: Binarize the separation result

    H = np.where(np.less(H, P), 0, W)
    P = np.where(np.greater_equal(H, P), 0, W)

    # Step 8: Generate separate waveforms
    H_temp = np.power(H, (1 / (2 * gamma))) * np.exp(1j * np.angle(F)) #ISTFT is taken first on this, with H
    P_temp = np.power(P, (1 / (2 * gamma))) * np.exp(1j * np.angle(F)) # ISTFT is taken second on this, with P

    _, h = istft(H_temp, fs=fs, window='hann', nperseg=winlen,
                                  noverlap=int(winlen/2), nfft=winlen, input_onesided=True)
    _, p = istft(P_temp, fs=fs, window='hann', nperseg=winlen,
                                    noverlap=int(winlen/2), nfft=winlen, input_onesided=True)

    #####################################################################################################
    plt.figure(1)
    plt.subplot(2, 1, 1)
    t_scale = np.linspace(0, len(h) / fs, len(h))
    plt.plot(t_scale, h)
    plt.title('Time domain visualization of h(t)')
    plt.axis('tight')
    plt.grid('on')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')

    plt.subplot(2, 1, 2)
    t_scale = np.linspace(0, len(p) / fs, len(p))
    plt.plot(t_scale, p)
    plt.title('Time domain visualization of p(t)')
    plt.axis('tight')
    plt.grid('on')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.show()

    write('./outputs/h_' + file_name, int(fs), np.int16(h))
    write('./outputs/p_' + file_name, int(fs), np.int16(p))

if __name__ == '__main__':
    print("Beginning run")
    separate_instruments("rhythm_birdland.wav")
    print("Completed")
