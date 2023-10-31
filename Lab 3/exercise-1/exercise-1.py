import numpy as np
import matplotlib.pyplot as plt
import math


def get_fourier_matrix(N: int) -> np.ndarray:
    matrix = np.ndarray((N, N), dtype=complex)

    for w in range(N):
        for n in range(N):
            matrix[w][n] = math.e ** (-2j * math.pi * n * w / N)

    return matrix


def plot_fourier_matrix(fourier_matrix: np.ndarray, title: str = None, file_types: [str] = None) -> None:
    figure, axes = plt.subplots(len(fourier_matrix), 1)
    figure.suptitle("Rows of Fourier matrix")
    figure.set_size_inches(25, 15)

    x_values = np.array([_ for _ in range(len(fourier_matrix))])
    for i, fourier_line in enumerate(fourier_matrix):
        cosine = np.array([value.real for value in fourier_line])
        sine = np.array([value.imag for value in fourier_line])

        axes[i].plot(x_values, cosine, label="re")
        axes[i].plot(x_values, sine, label="imag")
        axes[i].set_title(f"w={i}")
        axes[i].legend()

    if title and file_types:
        for file_type in file_types:
            plt.savefig(f"{title}.{file_type}")

    plt.show()


def main():
    fourier_matrix = get_fourier_matrix(8)
    plot_fourier_matrix(fourier_matrix, "exercise-1", ["png", "pdf"])

    transpose_complex_conjugate_fourier_matrix = np.conjugate(fourier_matrix.T)
    if np.allclose(np.dot(fourier_matrix, transpose_complex_conjugate_fourier_matrix), 8 * np.eye(8)):
        print("Fourier matrix is unitary.")
    else:
        print("Something bad had occurred.")


if __name__ == '__main__':
    main()
