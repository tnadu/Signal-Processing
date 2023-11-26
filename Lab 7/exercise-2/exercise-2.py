import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def get_attenuated_image(fft_of_original_image, frequency_cutoff):
    fft_db = 20 * np.log10(fft_of_original_image)
    fft_of_attenuated_image = fft_of_original_image.copy()
    fft_of_attenuated_image[fft_db > frequency_cutoff] = 0

    attenuated_image = np.real(np.fft.ifft2(fft_of_attenuated_image))
    return attenuated_image


def get_signal_to_noise_ratio(original_image, power_of_original_image, attenuated_image):
    power_of_noise = np.linalg.norm(original_image - attenuated_image) ** 2
    return 10 * np.log10(power_of_original_image / power_of_noise)


def show_attenuated_image(attenuated_image, frequency_cutoff, signal_to_noise_ratio):
    plt.figure(figsize=(12, 10))
    plt.imshow(attenuated_image, cmap=plt.cm.gray)
    plt.suptitle(f"Frequency cutoff = {frequency_cutoff}, SNR = {signal_to_noise_ratio:.3f}dB")
    plt.savefig(f"exercise-2-freq-cutoff-{frequency_cutoff}.png")
    plt.savefig(f"exercise-2-freq-cutoff-{frequency_cutoff}.pdf")
    plt.show()


def main():
    X = sp.datasets.face(gray=True)
    Y = np.fft.fft2(X)
    power_of_X = np.linalg.norm(X) ** 2

    for frequency_cutoff in [95, 100, 110, 120, 130, 140]:
        X_attenuated = get_attenuated_image(Y, frequency_cutoff)
        signal_to_noise_ratio = get_signal_to_noise_ratio(X, power_of_X, X_attenuated)
        show_attenuated_image(X_attenuated, frequency_cutoff, signal_to_noise_ratio)


if __name__ == '__main__':
    main()
