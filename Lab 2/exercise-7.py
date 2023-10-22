# a) the sampling frequency is artificially reduced to 250Hz.
# as long as it remains at least double the frequency of the
# signal, the signal itself is preserved. depending on the
# frequency of the signal, the greater the ratio of the sampling
# frequency to the sampled frequency, the less affected the plot
# becomes. when that ratio is low, some signal peaks get squashed
#
# b) the two down-sampled plots look fairly similar, with the
# exception that the first one seems out of phase by about 1s
import numpy as np
import matplotlib.pyplot as plt


def main():
    signal = lambda t: np.cos(2 * np.pi * 64 * t)
    discrete_time_interval = np.linspace(0, 1, 1000)

    figure, axes = plt.subplots(3, 1, layout="constrained")
    figure.suptitle(f"Down-sampling")

    axes[0].plot(discrete_time_interval, signal(discrete_time_interval))
    axes[0].set_title(f"Original sample")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("amplitude")

    axes[1].plot(discrete_time_interval[3::4], signal(discrete_time_interval)[3::4])
    axes[1].set_title(f"Down-sampled with a factor of 4, starting from the 4th sample")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("amplitude")

    axes[2].plot(discrete_time_interval[1::4], signal(discrete_time_interval)[1::4])
    axes[2].set_title(f"Down-sampled with a factor of 4, starting from the 2nd sample")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("amplitude")

    plt.show()


if __name__ == '__main__':
    main()
