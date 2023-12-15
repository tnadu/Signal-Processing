import numpy as np
import matplotlib.pyplot as plt


def generate_time_series(N):
    discrete_time_interval = np.linspace(0, 10, N)
    trend = lambda t: 0.5 * t ** 2
    seasonal = lambda t: 2 * np.sin(2 * np.pi * 8 * t) + 1.5 * np.cos(2 * np.pi * 4 * t)
    residuals = np.random.normal(scale=2, size=N)
    time_series = trend(discrete_time_interval) + seasonal(discrete_time_interval) + residuals

    return trend, seasonal, residuals, time_series, discrete_time_interval


def get_exponential_averaging(time_series, alpha):
    N = len(time_series)

    exponential_averaging = np.ndarray(N)
    exponential_averaging[0] = time_series[0]

    for i in range(1, N):
        exponential_averaging[i] = alpha * time_series[i] + (1 - alpha) * exponential_averaging[i - 1]

    return exponential_averaging


def get_optimum_alpha(time_series, precision=1e-2):
    optimum_alpha = 0
    optimum_score = np.inf

    for i in range(1, int(1 / precision) + 1):
        alpha = precision * i
        exponential_averaging = get_exponential_averaging(time_series, alpha)
        score = sum((time_series[1:] - exponential_averaging[:-1]) ** 2)

        if score < optimum_score:
            optimum_score = score
            optimum_alpha = alpha

    return optimum_alpha


def main():
    trend, seasonal, residuals, time_series, discrete_time_interval = generate_time_series(1000)
    optimum_alpha = get_optimum_alpha(time_series, 1e-3)
    exponential_averaging = get_exponential_averaging(time_series, optimum_alpha)
    print(f"Optimum alpha: {optimum_alpha:.3f}")

    figure, axes = plt.subplots(2, layout="constrained")
    axes[0].plot(time_series)
    axes[0].set_title("Generated Time Series")
    axes[1].plot(exponential_averaging)
    axes[1].set_title("Exponential Averaging")

    plt.savefig("exercise-2.png")
    plt.savefig("exercise-2.pdf")
    plt.show()


if __name__ == '__main__':
    main()
