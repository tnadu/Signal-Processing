import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def main():
    samples = np.genfromtxt("Train.csv", delimiter=",", skip_header=1, usecols=(2,))

    # 3 * 24h = 72h, Ts = 1h => 72 consecutive samples can be chosen
    chosen_samples = samples[24:96]

    # b)
    figure, axis = plt.subplots(2, 2, layout="constrained")
    figure.suptitle("Train.csv with a moving average filter applied")
    figure.set_size_inches(25, 14)

    for i, window_size in enumerate([5, 9, 13, 17]):
        filtered_samples = np.convolve(chosen_samples, np.ones(window_size), "valid") / window_size

        axis[i//2][i%2].plot(filtered_samples)
        axis[i//2][i%2].set_title(f"Window size = {window_size}")
        axis[i//2][i%2].set_xlabel("time (h)")
        axis[i//2][i%2].set_ylabel("number of cars")

    plt.savefig("exercise-4-figure-1.png")
    plt.savefig("exercise-4-figure-1.pdf")
    plt.show()

    # d)
    low_pass_threshold_frequency = 1 / (12 * 3600)
    b_butter, a_butter = sp.signal.butter(N=5, Wn=low_pass_threshold_frequency, fs = 1 / 3600)
    b_cheby, a_cheby = sp.signal.cheby1(N=5, rp=5, Wn=low_pass_threshold_frequency, fs = 1 / 3600)

    # e)
    # I would choose the Butterworth filter, because it seems to preserve the signal amplitudes much
    # more accurately, which is quite important, given that these are precise measurements, which
    # should remain as unaffected as possible
    butter_filtered_samples = sp.signal.filtfilt(b_butter, a_butter, chosen_samples)
    cheby_filtered_samples = sp.signal.filtfilt(b_cheby, a_cheby, chosen_samples)
    
    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Order of the filters = 5")
    figure.set_size_inches(25, 14)
    
    axis.plot(chosen_samples, label="Original signal")
    axis.plot(butter_filtered_samples, label="Butterworth-filtered")
    axis.plot(cheby_filtered_samples, label="Chebyshev-filtered")
    axis.set_xlabel("time (h)")
    axis.set_ylabel("number of cars")
    axis.legend()

    plt.savefig("exercise-4-figure-2.png")
    plt.savefig("exercise-4-figure-2.pdf")
    plt.show()

    # f)
    # testing a lower order
    b_butter, a_butter = sp.signal.butter(N=3, Wn=low_pass_threshold_frequency, fs=1 / 3600)
    b_cheby, a_cheby = sp.signal.cheby1(N=3, rp=5, Wn=low_pass_threshold_frequency, fs=1 / 3600)

    butter_filtered_samples = sp.signal.filtfilt(b_butter, a_butter, chosen_samples)
    cheby_filtered_samples = sp.signal.filtfilt(b_cheby, a_cheby, chosen_samples)

    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Order of the filters = 3")
    figure.set_size_inches(25, 14)

    axis.plot(chosen_samples, label="Original signal")
    axis.plot(butter_filtered_samples, label="Butterworth-filtered")
    axis.plot(cheby_filtered_samples, label="Chebyshev-filtered")
    axis.set_xlabel("time (h)")
    axis.set_ylabel("number of cars")
    axis.legend()

    plt.savefig("exercise-4-figure-3.png")
    plt.savefig("exercise-4-figure-3.pdf")
    plt.show()

    # testing a higher order
    b_butter, a_butter = sp.signal.butter(N=9, Wn=low_pass_threshold_frequency, fs=1 / 3600)
    b_cheby, a_cheby = sp.signal.cheby1(N=9, rp=5, Wn=low_pass_threshold_frequency, fs=1 / 3600)

    butter_filtered_samples = sp.signal.filtfilt(b_butter, a_butter, chosen_samples)
    cheby_filtered_samples = sp.signal.filtfilt(b_cheby, a_cheby, chosen_samples)

    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Order of the filters = 9")
    figure.set_size_inches(25, 14)

    axis.plot(chosen_samples, label="Original signal")
    axis.plot(butter_filtered_samples, label="Butterworth-filtered")
    axis.plot(cheby_filtered_samples, label="Chebyshev-filtered")
    axis.set_xlabel("time (h)")
    axis.set_ylabel("number of cars")
    axis.legend()

    plt.savefig("exercise-4-figure-4.png")
    plt.savefig("exercise-4-figure-4.pdf")
    plt.show()
    
    # testing various values for 'rp'
    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Order of the filters = 9")
    figure.set_size_inches(25, 14)

    axis.plot(chosen_samples, label="Original signal")
    for rp in [3, 5, 9, 11]:
        b_cheby, a_cheby = sp.signal.cheby1(N=9, rp=rp, Wn=low_pass_threshold_frequency, fs=1 / 3600)
        cheby_filtered_samples = sp.signal.filtfilt(b_cheby, a_cheby, chosen_samples)
        axis.plot(cheby_filtered_samples, label=f"Chebyshev-filtered-rp-{rp}")
    axis.set_xlabel("time (h)")
    axis.set_ylabel("number of cars")
    axis.legend()

    plt.savefig("exercise-4-figure-5.png")
    plt.savefig("exercise-4-figure-5.pdf")
    plt.show()


if __name__ == "__main__":
    main()
