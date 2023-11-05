import numpy as np
import matplotlib.pyplot as plt


def main():
    # f = 24, fs = 96, k = {0, -1, -2}
    aliased_signals = [lambda t: np.sin(2 * np.pi * (24 + 0 * 96) * t),
                       lambda t: np.sin(2 * np.pi * (24 + (-1) * 96) * t),
                       lambda t: np.sin(2 * np.pi * (24 + (-2) * 96) * t)]
    colors = ["mediumslateblue", "purple", "green"]
    discrete_time_interval = np.append(np.linspace(0, 1, 96, endpoint=False), 1)
    nyquist_discrete_time_interval = np.append(np.linspace(0, 1, 4096, endpoint=False), 1)

    amplitudes = []
    nyquist_amplitudes = []
    for aliased_signal in aliased_signals:
        amplitudes.append(aliased_signal(discrete_time_interval))
        nyquist_amplitudes.append(aliased_signal(nyquist_discrete_time_interval))

    figure, axes = plt.subplots(4, layout="constrained")
    figure.set_size_inches(20, 14)
    figure.suptitle("The aliasing phenomenon")

    axes[0].plot(nyquist_discrete_time_interval, nyquist_amplitudes[0], color=colors[0])
    for i in range(0, len(amplitudes)):
        axes[i+1].plot(nyquist_discrete_time_interval, nyquist_amplitudes[i], color=colors[i])
        axes[i+1].plot(discrete_time_interval, amplitudes[i], linestyle="", marker="o", color="indianred")

    plt.savefig("exercise-3.png")
    plt.savefig("exercise-3.pdf")
    plt.show()


if __name__ == '__main__':
    main()
