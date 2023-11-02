import time
import math
import numpy as np
import matplotlib.pyplot as plt


def get_discrete_fourier_transform(samples):
    n = len(samples)
    matrix = np.ndarray((n, n), dtype=complex)

    for m in range(n):
        for k in range(n):
            matrix[m][k] = math.e ** (-2j * math.pi * m * k / n)

    return np.dot(matrix, samples.T)


def main():
    signal = lambda t: np.cos(2 * np.pi * 6 * t) + np.cos(2 * np.pi * 36 * t) + np.cos(2 * np.pi * 120 * t) + np.cos(2 * np.pi * 150 * t)
    sample_numbers = [128, 256, 512, 1024, 2048, 4096, 8192]

    # fs = 384Hz
    execution_time_dft = np.ndarray((len(sample_numbers)))
    execution_time_fft = np.ndarray((len(sample_numbers)))

    print("Computing... (this may take a while)")
    for i, sample_number in enumerate(sample_numbers):
        discrete_time_interval = np.linspace(0, sample_number // 384, sample_number)
        amplitudes = signal(discrete_time_interval)

        start_time = time.time()
        get_discrete_fourier_transform(amplitudes)
        end_time = time.time()

        execution_time_dft[i] = round(end_time - start_time, 4)
        print(f"Finished DFT for N={sample_number} in {execution_time_dft[i]}s")

        start_time = time.time()
        np.fft.fft(amplitudes)
        end_time = time.time()

        execution_time_fft[i] = round(end_time - start_time, 4)
        print(f"Finished FFT for N={sample_number} in {execution_time_fft[i]}s")

    figure, axis = plt.subplots(1, layout="constrained")
    figure.set_size_inches(10, 7)
    figure.suptitle("Execution time of python implementation of DFT and of numpy implementation of FFT")

    axis.plot(sample_numbers, execution_time_dft, label="DFT")
    axis.plot(sample_numbers, execution_time_fft, label="FFT")
    axis.set_xlabel("Sample size")
    axis.set_ylabel("Execution time (s)")
    axis.legend()

    plt.yscale("log")
    plt.savefig("exercise-1.png")
    plt.savefig("exercise-1.pdf")
    plt.show()


if __name__ == '__main__':
    main()
