# a) the signal is bulging in the middle
# b) the signal is sunken in the middle
# c) there is no signal whatsoever
#
# a), b) The Nyquist-Shannon sampling theorem states that a signal must be sampled
# with a rate at least double the maximum signal frequency, in order to preserve it.
import numpy as np
import matplotlib.pyplot as plt


def main():
    sampling_frequency = 128

    signal = lambda t, frequency: np.sin(2 * np.pi * frequency * t)
    discrete_time_interval = np.linspace(0, 1, sampling_frequency)
    signal_frequencies = [sampling_frequency / 2, sampling_frequency / 4, 0]

    figure, axes = plt.subplots(3, 1, layout="constrained")
    figure.suptitle(f"Sampling frequency = {sampling_frequency}")
    for i, axis in enumerate(axes):
        axis.plot(discrete_time_interval, signal(discrete_time_interval, signal_frequencies[i]))
        axis.set_title(f"Frequency = {signal_frequencies[i]}")
        axis.set_xlabel("time")
        axis.set_ylabel("amplitude")

    plt.show()


if __name__ == '__main__':
    main()
