import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal import istft
from scipy.io.wavfile import read
from scipy.io import wavfile

def separate_instruments():

    # Read the file
    fs, x = read('rhythm_birdland.wav')

    # winlen_ms = 50
    # winlen = int(np.power(2, np.ceil(np.log2(float(winlen_ms) / 1000.0 * float(fs)))))
    winlen = 1024

    # Step 1: Generate STFT

    # Have used terminology from the paper
    # h and i represent indices of frequency and time bins
    h, i, F = stft(x=x, fs=fs, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
                   nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)

    # # Plot spectrogram
    # plt.pcolormesh(i, h, np.abs(F))
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # Step 2: Calculate a range-compressed version of the power spectrogram
    gamma = 0.3
    W = np.power(np.abs(F), 2 * gamma)

    # # Plot spectrogram
    # plt.pcolormesh(i, h, W)
    # plt.title('Power spectrogram')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # Step 3: Initialise
    k_max = 50
    H = P = 0.5 * W
    alpha = 0.3

    zero_np=np.empty_like(H) #A zero array
    print(zero_np)

    for k in range(k_max):
        # H_new = np.empty_like(H)
        # P_new = np.empty_like(P)
        delta = term_1 = term_2 = np.empty_like(H)

        # Step 4: Calculate update variable delta
        for i_iter in range(1, np.shape(H)[1]-1):
            term_1[:, i_iter] = alpha * ((H[:, i_iter-1] + H[:, i_iter+1] - (2 * H[:, i_iter])) / 4)
        term_1[:, 0] = alpha * ((H[:, 1] - H[:, 0]) / 2)
        term_1[:, -1] = alpha * ((H[:, -2] - H[:, -1]) / 2)

        for h_iter in range(1, np.shape(H)[0]-1):
            term_2[h_iter, :] = (1 - alpha) * ((P[h_iter-1, :] + P[h_iter+1, :] - (2 * P[h_iter, :])) / 4)
        term_2[0, :] = (1 - alpha) * ((P[1, :] - P[0, :]) / 2)
        term_2[-1, :] = (1 - alpha) * ((P[-2, :] - P[-1, :]) / 2)

        delta = term_1 - term_2

        # Step 5: Update H and P
        H=np.minimum(np.maximum(H+delta,zero_np),W)
        P=W - H

        # Step 6: Increment k (automatically through loop)

    # Step 7: Binarize the separation result

    #if (H<P).all():
       # H=np.empty_like(H)*0
     #   P=W
   # else:
       # H=W
       # P=np.empty_like(H)*0

    H=np.where((H<P).all(),zero_np,W)
    P= np.where((H >= P).all(), W, zero_np)

    # Step 8: Generate separate waveforms

    first_function= np.power(H,(1/(2*gamma)))*  np.exp(np.angle(F)) #ISTFT is taken first on this, with H
    second_function = np.power(P, (1 / (2 * gamma))) * np.exp(np.angle(F)) # ISTFT is taken second on this, with P

    _,output_one = istft(first_function,fs=fs,window='hann',nperseg=winlen,noverlap=winlen/2,nfft=winlen,input_onesided=True)
    _,output_two = istft(second_function, fs=fs, window='hann', nperseg=winlen,noverlap=winlen / 2, nfft=winlen,input_onesided=True)

#####################################################################################################
    plt.figure(1)
    plt.subplot(2, 1, 1)
    t_scale = np.linspace(0, len(output_one) / fs, len(output_one))
    plt.plot(t_scale, output_one)
    plt.title('Time domain visualization of h(t)')
    plt.axis('tight')
    plt.grid('on')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')

    plt.subplot(2, 1, 2)
    t_scale = np.linspace(0, len(output_two) / fs, len(output_two))
    plt.plot(t_scale, output_two)
    plt.title('Time domain visualization of p(t)')
    plt.axis('tight')
    plt.grid('on')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.show()

    print(delta)
    wavfile.write('h(t).wav', int(fs), np.int16(output_one))
    wavfile.write('p(t).wav', int(fs), np.int16(output_two))

if __name__ == '__main__':
    print("Beginning run")
    separate_instruments()
    print("Completed")
