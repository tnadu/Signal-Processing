import numpy as np
import matplotlib.pyplot as plt


def get_hanning_window(size: int) -> np.ndarray[float]:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / size))


def get_rectangular_window(size: int) -> np.ndarray[float]:
    return np.ones(size)


def main():
    signal = lambda t: np.sin(2 * np.pi * 100 * t)
    discrete_time_interval = np.linspace(0, 1, 1000)

    figure, axes = plt.subplots(1, 2, layout="constrained")
    figure.set_size_inches(25, 14)

    hanning_windowed_signal = np.multiply(signal(discrete_time_interval), get_hanning_window(1000))
    axes[0].plot(discrete_time_interval, hanning_windowed_signal)
    axes[0].set_title("Hanning window applied over the signal")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")

    rectangular_windowed_signal = np.multiply(signal(discrete_time_interval), get_rectangular_window(1000))
    axes[1].plot(discrete_time_interval, rectangular_windowed_signal)
    axes[1].set_title("Rectangular window applied over the signal")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amplitude")

    plt.savefig("exercise-3.png")
    plt.savefig("exercise-3.pdf")
    plt.show()


if __name__ == "__main__":
    main()
