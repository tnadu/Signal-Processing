import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def x(t, phase = 0):
    return np.sin(80 * np.pi * t + phase)


def compute_gamma(samples, noise, snr):
    return np.sqrt(np.linalg.norm(samples, ord=2) ** 2 / (snr * np.linalg.norm(noise, ord=2) ** 2))


def show_as_subplots_by_phase(discrete_time_interval: np.ndarray, signal: Callable[[float | np.ndarray, float], float | np.ndarray], phases: [float], title: str) -> None:
    figure, axes = plt.subplots(len(phases), 1, )
    figure.suptitle(title)

    for i, phase in enumerate(phases):
        axes[i].plot(discrete_time_interval, signal(discrete_time_interval, phase))
        axes[i].set_xlabel("time")
        axes[i].set_ylabel("amplitude")

    plt.show()


def show_as_subplots_by_gamma(discrete_time_interval: np.ndarray, signal: Callable[[float | np.ndarray, float], float | np.ndarray], noise: np.ndarray, gammas: np.ndarray, title: str) -> None:
    figure, axes = plt.subplots(len(gammas), 1, )
    figure.suptitle(title)

    for i, gamma in enumerate(gammas):
        noisy_signal = signal(discrete_time_interval) + gamma * noise
        axes[i].plot(discrete_time_interval, noisy_signal)
        axes[i].set_xlabel("time")
        axes[i].set_ylabel("amplitude")

    plt.show()


def main():
    discrete_time_interval = np.linspace(0, 1, 3200)
    phases = [np.pi / 6, np.pi / 12, np.pi/3, np.pi / 5]
    show_as_subplots_by_phase(discrete_time_interval, x, phases, "The same sinusoidal with different phases")

    noise = np.random.normal(size=3200)
    snrs = [0.1, 1, 10, 100]
    gammas = [compute_gamma(x(discrete_time_interval), noise, snr) for snr in snrs]
    show_as_subplots_by_gamma(discrete_time_interval, x, noise, gammas, "Different NSRs")


if __name__ == '__main__':
    main()
