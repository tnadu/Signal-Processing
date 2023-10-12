import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


def x(t: float | np.ndarray) -> float | np.ndarray:
    return np.cos(520 * np.pi * t + np.pi / 3)


def y(t: float | np.ndarray) -> float | np.ndarray:
    return np.cos(280 * np.pi * t - np.pi / 3)


def z(t: float | np.ndarray) -> float | np.ndarray:
    return np.cos(120 * np.pi * t + np.pi / 3)


def show_as_subplots(discrete_time_interval: np.ndarray, signals: [Callable[[float | np.ndarray], float | np.ndarray]], title: str) -> None:
    figure, axes = plt.subplots(len(signals), 1, )
    figure.suptitle(title)

    for i, signal in enumerate(signals):
        axes[i].stem(discrete_time_interval, signal(discrete_time_interval))
        axes[i].set_xlabel("time")
        axes[i].set_ylabel("amplitude")
        
    plt.show()


def main():
    signals = [x, y, z]

    discrete_time_interval = np.linspace(0, 0.03, 60)
    print(f"a) {discrete_time_interval}\n")

    print("b) showing...")
    show_as_subplots(discrete_time_interval, signals, "2000Hz - [0:0.0005:0.03]")

    print("c) showing...")
    show_as_subplots(np.linspace(0, 0.03, 6), signals, "200Hz - [0:0.005:0.03]")


if __name__ == '__main__':
    main()
