import numpy as np
import sounddevice


def main():
    discrete_time_interval = np.linspace(0, 5, 220500)
    first_signal = lambda t: 3 * np.cos(2 * np.pi * 480 * t)
    second_signal = lambda t: 3 * np.cos(2 * np.pi * 640 * t)

    # the second signal sounds higher pitched
    sounddevice.play(np.concatenate((first_signal(discrete_time_interval), second_signal(discrete_time_interval))), samplerate=44100)
    sounddevice.wait()


if __name__ == '__main__':
    main()
