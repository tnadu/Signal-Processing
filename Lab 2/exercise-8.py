import numpy as np
import matplotlib.pyplot as plt


def main():
    discrete_interval = np.linspace(-np.pi / 2, np.pi / 2, 128)

    figure, axis = plt.subplots(layout="constrained")
    figure.suptitle("sin(x) ~= x approximation")
    axis.plot(discrete_interval, np.sin(discrete_interval), label="sin(x)")
    axis.plot(discrete_interval, discrete_interval, label="x")
    axis.set_xlabel("angle")
    axis.set_ylabel("value")
    axis.legend()
    plt.show()

    figure, axis = plt.subplots(layout="constrained")
    figure.suptitle("Error between sin(x) and x")
    axis.plot(discrete_interval, np.abs(np.sin(discrete_interval) - discrete_interval))
    axis.set_xlabel("angle")
    axis.set_ylabel("error")
    plt.show()

    pade_approximation = lambda x: (x - 7 * x ** 3 / 60) / (1 + x ** 2 / 20)

    figure, axis = plt.subplots(layout="constrained")
    figure.suptitle("Pade approximation")
    axis.plot(discrete_interval, np.sin(discrete_interval), label="sin(x)")
    axis.plot(discrete_interval, pade_approximation(discrete_interval), label="pade(x)")
    axis.set_xlabel("angle")
    axis.set_ylabel("value")
    axis.legend()
    plt.show()

    figure, axis = plt.subplots(layout="constrained")
    figure.suptitle("Error between sin(x) and Pade approximation")
    axis.plot(discrete_interval, np.abs(np.sin(discrete_interval) - pade_approximation(discrete_interval)))
    axis.set_xlabel("angle")
    axis.set_ylabel("error")
    plt.show()


if __name__ == '__main__':
    main()
