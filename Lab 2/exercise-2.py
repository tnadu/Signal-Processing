import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any


def x(t: float | np.ndarray, phase: float = 0) -> float | np.ndarray:
    return np.sin(2 * np.pi * 8 * t + phase)


def compute_gamma(samples: np.ndarray, noise: np.ndarray, signal_to_noise_ratio: float) -> float:
    return np.sqrt(np.linalg.norm(samples, ord=2) ** 2 / (signal_to_noise_ratio * np.linalg.norm(noise, ord=2) ** 2))


def plot_by_phase(axis: Any, signal: Callable[[float | np.ndarray, float], float | np.ndarray], phases: [(float, str)],
                  discrete_time_interval: np.ndarray, noise: np.ndarray = None, gammas: [float] = None, snr: float = None):
    for i, phase in enumerate(phases):
        samples = signal(discrete_time_interval, phase[0])

        if noise is not None and gammas is not None and snr:
            samples += gammas[i] * noise

        axis.plot(discrete_time_interval, samples, label=phase[1])

    if noise is not None and gammas is not None and snr:
        axis.set_title(f"SNR={snr}")

    axis.set_xlabel("time")
    axis.set_ylabel("amplitude")
    axis.legend()


def main():
    discrete_time_interval = np.linspace(0, 1, 640)
    phases = [(np.pi/3, "pi/3"), (np.pi / 5, "pi/5"), (np.pi / 6, "pi/6"), (np.pi / 12, "pi/12")]

    figure, axis = plt.subplots(layout="constrained")
    figure.suptitle("Signals with different phases for the same amplitude and frequency")
    plot_by_phase(axis, x, phases, discrete_time_interval)
    plt.show()

    noise = np.random.normal(size=640)
    signal_to_noise_ratios = [0.1, 1, 10, 100, 10000]

    figure, axes = plt.subplots(len(signal_to_noise_ratios), 1, layout="constrained")
    figure.suptitle("The same signals with noise of various SNRs")
    for i, signal_to_noise_ratio in enumerate(signal_to_noise_ratios):
        gammas = [compute_gamma(x(discrete_time_interval, phase[0]), noise, signal_to_noise_ratio) for phase in phases]
        plot_by_phase(axes[i], x, phases, discrete_time_interval, noise, gammas, signal_to_noise_ratio)
    plt.show()


if __name__ == '__main__':
    main()
