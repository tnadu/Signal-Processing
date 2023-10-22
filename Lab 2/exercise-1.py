import numpy as np
import matplotlib.pyplot as plt


def sin_sinusoidal(t: float | np.ndarray) -> float | np.ndarray:
    return 3 * np.sin(2 * np.pi * 20 * t)


def cos_sinusoidal(t: float | np.ndarray) -> float | np.ndarray:
    return 3 * np.cos(2 * np.pi * 20 * t - np.pi / 2)


def main():
    discrete_time_interval = np.linspace(0, 1, 1200)

    figure, axes = plt.subplots(2, 1, )
    figure.suptitle("Identical sin and cos signals")
    axes[0].plot(discrete_time_interval, sin_sinusoidal(discrete_time_interval))
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("amplitude")

    axes[1].plot(discrete_time_interval, cos_sinusoidal(discrete_time_interval))
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("amplitude")

    plt.show()


if __name__ == '__main__':
    main()
