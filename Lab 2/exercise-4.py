import numpy as np
import matplotlib.pyplot as plt


def main():
    discrete_time_interval = np.linspace(0, 1, 800)
    sinusoidal = lambda t: 2 * np.sin(2 * np.pi * 16 * t - np.pi / 2)
    sawtooth = lambda t: 2 * (20 * t - np.floor(20 * t))

    figure, axes = plt.subplots(3, 1, layout="constrained")
    figure.suptitle("A sinusoidal and a sawtooth signal and their sum")

    axes[0].plot(discrete_time_interval, sinusoidal(discrete_time_interval))
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("amplitude")

    axes[1].plot(discrete_time_interval, sawtooth(discrete_time_interval))
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("amplitude")

    axes[2].plot(discrete_time_interval, sinusoidal(discrete_time_interval) + sawtooth(discrete_time_interval))
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("amplitude")

    plt.show()


if __name__ == '__main__':
    main()
