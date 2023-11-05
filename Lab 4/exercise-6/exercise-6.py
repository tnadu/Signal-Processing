import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile


def main():
    sample_rate, samples = scipy.io.wavfile.read('../exercise-5/vowels.wav')

    # as I understood it, we must divide the total amount of samples into groups of size 1%,
    # where each group is made up of two halves (.5%-sized each). these smaller bins, which
    # make up each group, are shared between neighbouring groups, so that when a bin is
    # encountered, it is initially the second half of a given group, and then the first half
    # of the next group.

    # getting the index of the first sample in each group (groups overlap - we use increments of .5%)
    group_indices = np.linspace(0, len(samples), 200, endpoint=False).astype(int)
    # 1% of the number of samples
    group_size = len(samples) // 100
    # we take group_size - 1 samples after the first sample in each group
    groups = [samples[i:i+group_size] for i in group_indices[:199]]

    # applying FFT over each group, discarding the second half of each FT, since it is mirrored
    fast_fourier_transforms = [np.fft.fft(group)[:group_size // 2] for group in groups]
    # taking the absolute value of each FT and transposing the resulting matrix, so that the
    # values of each FT end up on a column, not a row
    spectrogram_matrix = np.absolute(np.array(fast_fourier_transforms)).T

    # getting the temporal information (n/fs = nTs)
    discrete_time_interval = np.arange(0, len(samples), 1) / sample_rate
    # only the moment corresponding to the first sample in each group will be considered
    discrete_time_interval = discrete_time_interval[group_indices[:199]]
    # each frequency bin n corresponds to the frequency n * fs / N
    fft_frequencies = np.arange(0, group_size // 2, 1) * sample_rate / group_size

    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Spectrogram of recording of vowels")
    figure.set_size_inches(20, 6)

    image = axis.contourf(discrete_time_interval, fft_frequencies, spectrogram_matrix, cmap="plasma")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    figure.colorbar(image)

    plt.savefig("exercise-6.png")
    plt.savefig("exercise-6.pdf")
    plt.show()


if __name__ == '__main__':
    main()
