import numpy as np
import matplotlib.pyplot as plt
import math


def get_signal_winding(amplitudes, time, frequency=1.0):
    winding = np.ndarray((len(amplitudes)), dtype=complex)

    for i in range(len(amplitudes)):
        winding[i] = amplitudes[i] * math.e ** (-2j * math.pi * frequency * time[i])

    return winding


def plot_winding(axis, real_coordinates, imaginary_coordinates, color, center=None, stem_sample=None, title=None):
    axis.plot(real_coordinates, imaginary_coordinates, color=color)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.axhline(color="black", linewidth=1)
    axis.axvline(color="black", linewidth=1)
    axis.set_xlabel("Real")
    axis.set_ylabel("Imaginary")

    if stem_sample:
        axis.plot([real_coordinates[stem_sample], 0], [imaginary_coordinates[stem_sample], 0], color="red", marker="o", markevery=2)

    if center:
        axis.plot([center.real, 0], [center.imag, 0], color="black", marker="o", markevery=2, linewidth=3)

    if title:
        axis.set_title(title)


def get_center(winding):
    return np.sum(winding) / len(winding)


def get_color(center):
    red = np.max(0.65 - np.linalg.norm([center.real, center.imag]) / 2, 0)
    green = 0
    blue = 0.6

    return np.array([red, green, blue])


def main():
    signal = lambda t: np.cos(2 * np.pi * 9 * t)
    discrete_time_interval = np.linspace(0, 1, 1000)
    amplitudes = signal(discrete_time_interval)

    # first figure
    signal_winding = get_signal_winding(amplitudes, discrete_time_interval)
    figure, axes = plt.subplots(1, 2, layout="constrained")
    figure.set_size_inches(25, 15)
    figure.suptitle("Representation of a signal in the complex plane")

    axes[0].plot(discrete_time_interval * 1000, amplitudes, color="green")
    axes[0].stem(550, amplitudes[550], linefmt="red", markerfmt="red")
    axes[0].axhline(color="black", linewidth=1)
    axes[0].set_xlabel("Time (samples)")
    axes[0].set_ylabel("Amplitude")

    plot_winding(axes[1], signal_winding.real, signal_winding.imag, color="mediumslateblue", stem_sample=550)

    plt.savefig("exercise-2-figure-1.png")
    plt.savefig("exercise-2-figure-1.pdf")
    plt.show()

    # second figure
    frequencies = [[1, 3.14], [9, 13]]
    figure, axes = plt.subplots(2, 2, layout="constrained")
    figure.set_size_inches(15, 15)
    figure.suptitle("Representation of the Fourier Transform in the complex plane")
    for i in range(len(frequencies)):
        for j in range(len(frequencies)):
            signal_winding = get_signal_winding(amplitudes, discrete_time_interval, frequency=frequencies[i][j])
            center = get_center(signal_winding)
            color = get_color(center)
            plot_winding(axes[i][j], signal_winding.real, signal_winding.imag, color=color, center=center, title=f"w = {frequencies[i][j]}")

    plt.savefig("exercise-2-figure-2.png")
    plt.savefig("exercise-2-figure-2.pdf")
    plt.show()


if __name__ == '__main__':
    main()
