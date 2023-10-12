import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def plot_one_dimensional_signal(discrete_time_interval: np.ndarray, signal: [Callable[[float | np.ndarray], float | np.ndarray]], title: str, exercise: str) -> None:
    sampled_values = signal(discrete_time_interval)
    print(f"{exercise}) showing...")
    plt.plot(discrete_time_interval, sampled_values)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.show()


def plot_two_dimensional_signal(signal: np.ndarray, title: str, exercise: str) -> None:
    print(f"{exercise}) showing...")
    plt.imshow(signal)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    signal = lambda t: np.sin(2 * np.pi * 400 * t)
    discrete_time_interval = np.linspace(0, 0.05, 1600)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Sinusoidal signal with a frequency of 400Hz, sampled with a frequency of 32000Hz (or 1600 samples in .05s)", "a")

    signal = lambda t: np.sin(2 * np.pi * 800 * t)
    discrete_time_interval = np.linspace(0, 3, 192000)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Sinusoidal signal with a frequency of 800Hz, sampled with a frequency of 64000Hz (or 192000 samples in 3s)", "b")

    signal = lambda t: 240 * t - np.floor(240 * t)
    discrete_time_interval = np.linspace(0, 1, 19200)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Sawtooth signal with a frequency of 240Hz, sampled with a frequency of 19200Hz", "c")

    signal = lambda t: np.sign(np.sin(2 * np.pi * 300 * t))
    discrete_time_interval = np.linspace(0, 1, 20000)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Square signal with a frequency of 300Hz, sampled with a frequency of 20000Hz", "d")

    signal = np.random.rand(2160, 1440)
    plot_two_dimensional_signal(signal, "Random signal", "e")

    signal = np.random.rand(128, 128)
    for i in range(128):
        signal[i] *= i
        signal[i] - np.random.rand() * i
    plot_two_dimensional_signal(signal, "Custom signal", "f")


if __name__ == '__main__':
    main()
