import numpy as np
import matplotlib.pyplot as plt


def main():
    aliased_signals = [lambda t: np.sin(2 * np.pi * (2 + 2 * 11) * t),
                       lambda t: np.sin(2 * np.pi * (2 + 1 * 11) * t),
                       lambda t: np.sin(2 * np.pi * (2 + 0 * 11) * t)]
    colors = ["mediumslateblue", "purple", "green"]
    discrete_time_interval = np.append(np.linspace(0, 1, 11, endpoint=False), 1)
    nyquist_discrete_time_interval = np.append(np.linspace(0, 1, 512, endpoint=False), 1)

    amplitudes = []
    nyquist_amplitudes = []
    for aliased_signal in aliased_signals:
        amplitudes.append(aliased_signal(discrete_time_interval))
        nyquist_amplitudes.append(aliased_signal(nyquist_discrete_time_interval))

    figure, axes = plt.subplots(4, layout="constrained")
    figure.set_size_inches(10, 7)
    figure.suptitle("The aliasing phenomenon")

    axes[0].plot(nyquist_discrete_time_interval, nyquist_amplitudes[0], color=colors[0])
    for i in range(0, len(amplitudes)):
        axes[i+1].plot(nyquist_discrete_time_interval, nyquist_amplitudes[i], color=colors[i])
        axes[i+1].plot(discrete_time_interval, amplitudes[i], linestyle="", marker="o", color="indianred")

    plt.savefig("exercise-2.png")
    plt.savefig("exercise-2.pdf")
    plt.show()


if __name__ == '__main__':
    main()
