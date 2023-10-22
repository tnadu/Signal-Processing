import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
import sounddevice
from typing import Callable


def plot_signal(discrete_time_interval: np.ndarray, signal: [Callable[[float | np.ndarray], float | np.ndarray]], title: str) -> None:
    sampled_values = signal(discrete_time_interval)
    plt.plot(discrete_time_interval, sampled_values)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.show()


def main():
    discrete_time_interval = np.linspace(0, 10, 441000)

    signal = lambda t: np.sin(2 * np.pi * 400 * t)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_signal(discrete_time_interval, signal, "a) Sinusoidal signal with a frequency of 400Hz, sampled with a frequency of 44100Hz")
    sounddevice.stop()

    signal = lambda t: np.sin(2 * np.pi * 800 * t)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_signal(discrete_time_interval, signal, "b) Sinusoidal signal with a frequency of 800Hz, sampled with a frequency of 44100Hz")
    sounddevice.stop()

    signal = lambda t: 240 * t - np.floor(240 * t)
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_signal(discrete_time_interval, signal, "c) Sawtooth signal with a frequency of 240Hz, sampled with a frequency of 44100Hz")
    sounddevice.stop()

    signal = lambda t: np.sign(np.sin(2 * np.pi * 300 * t))
    sounddevice.play(signal(discrete_time_interval), samplerate=44100)
    plot_signal(discrete_time_interval, signal, "d) Square signal with a frequency of 300Hz, sampled with a frequency of 44100Hz")
    sounddevice.stop()

    print("saving d) to the disk as 'square-300hz.wav'...")
    discrete_time_interval = np.linspace(0, 10, int(10e6))
    scipy.io.wavfile.write('square-300hz.wav', int(10e5), signal(discrete_time_interval))

    print("loading 'square-300hz.wav' from the disk...")
    rate, read_signal = scipy.io.wavfile.read('square-300hz.wav')
    print("loaded.")


if __name__ == '__main__':
    main()
