import numpy as np
import matplotlib.pyplot as plt
import math


def get_discrete_fourier_transform(samples):
    n = len(samples)
    matrix = np.ndarray((n, n), dtype=complex)

    for m in range(n):
        for k in range(n):
            matrix[m][k] = math.e ** (-2j * math.pi * m * k / n)

    return np.dot(matrix, samples.T)


def main():
    signal = lambda t: np.cos(2 * np.pi * 12 * t) + 2 * np.cos(2 * np.pi * 60 * t) + np.cos(2 * np.pi * 132 * t) + 0.5 * np.cos(2 * np.pi * 516 * t)
    # sampling frequency must be at least twice the largest frequency in the signal, in order not to lose information
    discrete_time_interval = np.linspace(0, 1, 2064)
    amplitudes = signal(discrete_time_interval)

    # computed winding frequencies will be [0:1032:6] Hz (starting from 0, up to 1032, with a step of 6Hz)
    frequency_bin = 6
    # N = q * fs; q = 1 / frequency_bin, fs = 2064Hz => N = 2064 / frequency_bin
    number_of_samples = 2064 // frequency_bin
    samples_used_in_fourier_transform = amplitudes[:number_of_samples]
    discrete_fourier_transform = get_discrete_fourier_transform(samples_used_in_fourier_transform)
    # only the first half of the Fourier transform is usable, since the second is mirrored
    usable_range = number_of_samples // 2
    # get the modulus of each DFT component, which are returned as complex numbers
    discrete_fourier_transform = np.absolute(discrete_fourier_transform[:usable_range])
    frequencies = [frequency_bin * i for i in range(usable_range)]

    figure, axes = plt.subplots(1, 2, layout="constrained")
    figure.set_size_inches(25, 15)
    figure.suptitle("Fourier Transform of a signal with 4 frequency components")

    axes[0].plot(discrete_time_interval, amplitudes, color="blue")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("x(t)")

    axes[1].stem(frequencies, discrete_fourier_transform, linefmt="black", markerfmt="black")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("|X(w)|")

    plt.savefig("exercise-3.png")
    plt.savefig("exercise-3.pdf")
    plt.show()


if __name__ == '__main__':
    main()
