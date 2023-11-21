import numpy as np
import matplotlib.pyplot as plt


def main():
    signal = np.random.rand(100)
    convoluted_signal = signal

    figure, axes = plt.subplots(2, 2, layout="constrained")
    figure.suptitle("Repeated convolutions of a randomly generated signal")
    figure.set_size_inches(20, 15)

    axes[0][0].plot(signal)
    axes[0][0].set_title("Original signal")

    for i in range(3):
        convoluted_signal = np.convolve(convoluted_signal, signal, mode="full")

        axes[(i+1)//2][(i+1)%2].plot(convoluted_signal)
        axes[(i+1)//2][(i+1)%2].set_title(f"Iteration #{i+1}")

    plt.savefig("exercise-1.png")
    plt.savefig("exercise-1.pdf")
    plt.show()


if __name__ == "__main__":
    main()
