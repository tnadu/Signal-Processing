import numpy as np


def main():
    N = 65

    size_of_p = np.random.randint(low=1, high=N, size=1)
    size_of_q = np.random.randint(low=1, high=N, size=1)

    p = np.random.randint(low=0, high=30, size=size_of_p)
    q = np.random.randint(low=0, high=30, size=size_of_q)
    print(f"p = {p}\n")
    print(f"q = {q}\n")

    multiplication_by_convolution = np.convolve(p, q, mode="full")
    print(f"p * q by convolution = \n{multiplication_by_convolution}\n")

    padded_p = np.append(p, np.zeros(size_of_q-1))
    padded_q = np.append(q, np.zeros(size_of_p-1))

    P = np.fft.fft(padded_p)
    Q = np.fft.fft(padded_q)

    multiplication_by_fft_product = np.fft.ifft(np.multiply(P, Q)).real
    print(f"p * q by FFT pointwise multiplication = \n{multiplication_by_fft_product}\n")


if __name__ == "__main__":
    main()
