import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io.wavfile import read


def separate_instruments(file_name = "./inputs/rhythm_birdland.wav"):

    # Read the file
    fs, x = read(file_name)

    # winlen_ms = 50
    # winlen = int(np.power(2, np.ceil(np.log2(float(winlen_ms) / 1000.0 * float(fs)))))
    winlen = 1024

    # Step 1: Generate STFT
    h, i, F = stft(x=x, fs=fs, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
                   nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)

    # Plot spectrogram
    plt.pcolormesh(i, h, np.abs(F))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # Step 2: Calculate a range-compressed version of the power spectrogram
    gamma = 0.3
    W = np.power(np.abs(F), 2 * gamma)

    # Plot spectrogram
    plt.pcolormesh(i, h, W)
    plt.title('Power spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()




if __name__ == '__main__':
    print("Beginning run")
    separate_instruments("./inputs/rhythm_birdland.wav")
    print("Completed")