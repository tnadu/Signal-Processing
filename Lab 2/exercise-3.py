import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
import sounddevice
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
    discrete_time_interval = np.linspace(0, 10, 441000)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Sinusoidal signal with a frequency of 400Hz, sampled with a frequency of 32000Hz (or 1600 samples in .05s)", "a")
    sounddevice.stop()

    signal = lambda t: np.sin(2 * np.pi * 800 * t)
    discrete_time_interval = np.linspace(0, 10, 441000)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Sinusoidal signal with a frequency of 800Hz, sampled with a frequency of 64000Hz (or 192000 samples in 3s)", "b")
    sounddevice.stop()

    signal = lambda t: 240 * t - np.floor(240 * t)
    discrete_time_interval = np.linspace(0, 10, 441000)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Sawtooth signal with a frequency of 240Hz, sampled with a frequency of 19200Hz", "c")
    sounddevice.stop()

    signal = lambda t: np.sign(np.sin(2 * np.pi * 300 * t))
    discrete_time_interval = np.linspace(0, 10, 441000)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_one_dimensional_signal(discrete_time_interval, signal, "Square signal with a frequency of 300Hz, sampled with a frequency of 20000Hz", "d")
    sounddevice.stop()

    print("saving d) to the disk as 'square-300hz.wav'...")
    discrete_time_interval = np.linspace(0, 10, int(10e6))
    scipy.io.wavfile.write('square-300hz.wav', int(10e5), signal(discrete_time_interval))

    print("loading 'square-300hz.wav' from the disk,,,")
    rate, read_signal = scipy.io.wavfile.read('square-300hz.wav')


if __name__ == '__main__':
    main()
