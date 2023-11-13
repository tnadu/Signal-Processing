import math
import numpy as np
import matplotlib.pyplot as plt


def main():
    # d)
    samples = np.genfromtxt("Train.csv", delimiter=",", skip_header=1, usecols=(2,))
    # getting rid of the DC offset (the continuous frequency component)
    samples -= np.mean(samples)
    sampling_frequency = 1 / 3600
    frequency_bin = sampling_frequency / len(samples)
    frequencies = np.arange(0, len(samples) // 2, 1) * frequency_bin

    fft = np.fft.fft(samples)[:len(samples) // 2]
    fft = np.absolute(fft)

    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("FFT of Train.csv")
    figure.set_size_inches(25, 15)

    axis.plot(frequencies, fft)
    axis.set_xlabel("Frequencies")
    axis.set_ylabel("|X(w)|")

    plt.savefig("exercise-1-figure-1.png")
    plt.savefig("exercise-1-figure-1.pdf")
    plt.show()

    # f)
    main_frequencies = np.argpartition(fft, -4)[-4:] * frequency_bin
    main_periods = 1 / main_frequencies / 3600 / 24
    print(f"The four main frequencies of the signal: {main_frequencies}")
    print(f"They correspond to the following periods, expressed in days: {main_periods}\n")

    # g)
    # sample 1000 corresponds to the moment 1000h (41d16h) after the point of start (00:00-25.08.2012),
    # which is 16:00-05.10.2012;
    # the next Monday is at 00:00-08.10.2012, 2d8h (56h) after sample 1000, which corresponds to sample 1056;
    # a month is considered to be equal to 30d (720h), which corresponds to sample 1776
    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("A month of traffic samples")
    figure.set_size_inches(25, 15)

    axis.plot(list(range(1056, 1776)), samples[1056:1776])
    axis.set_xlabel("Samples")
    axis.set_ylabel("Number of cars")

    plt.savefig("exercise-1-figure-2.png")
    plt.savefig("exercise-1-figure-2.pdf")
    plt.xticks(list(range(1056, 1776, 24)), [f"Day {i}" for i in range(1, 31)])
    plt.show()

    # i)
    # there shouldn't be significant frequency components with a period less than 12h
    # (roughly speaking, the difference between night and day), so we can discard them
    frequency_limit = 1 / (12 * 3600)   # frequency for a period of 12h
    # the first half is mirrored in the second one, so both sections around the center should be nulled
    start_frequency_bin_index = math.ceil(frequency_limit / frequency_bin)
    end_frequency_bin_index = len(samples) - start_frequency_bin_index

    fft = np.fft.fft(samples)
    fft[start_frequency_bin_index:end_frequency_bin_index+1] = 0j
    filtered_samples = np.fft.ifft(fft)

    figure, axis = plt.subplots(1, layout="constrained")
    figure.set_size_inches(25, 15)
    axis.plot(filtered_samples)
    axis.set_xlabel("Samples")
    axis.set_ylabel("Number of cars")

    plt.show()


if __name__ == '__main__':
    main()
