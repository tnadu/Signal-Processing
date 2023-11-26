import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def get_signal_to_noise_ratio(reference_image, power_of_reference_image, current_image):
    power_of_noise = np.linalg.norm(reference_image - current_image) ** 2
    return 10 * np.log10(power_of_reference_image / power_of_noise)


def show_denoised_image(denoised_image, filter_size, signal_to_noise_ratio):
    plt.figure(figsize=(12, 10))
    plt.imshow(denoised_image, cmap=plt.cm.gray)
    plt.suptitle(f"Uniform filter size = {filter_size}, SNR = {signal_to_noise_ratio:.3f}dB")
    plt.savefig(f"exercise-3-filter-size-{filter_size}.png")
    plt.savefig(f"exercise-3-filter-size-{filter_size}.pdf")
    plt.show()


def main():
    X = sp.datasets.face(gray=True)
    power_of_X = np.linalg.norm(X) ** 2

    noise = np.random.randint(-200, high=201, size=X.shape)
    X_noisy = X + noise

    plt.figure(figsize=(12, 10))
    plt.imshow(X_noisy, cmap=plt.cm.gray)
    plt.suptitle(f"Initial noisy image, SNR = {get_signal_to_noise_ratio(X, power_of_X, X_noisy):.3f}dB")
    plt.savefig("exercise-3-noisy.png")
    plt.savefig("exercise-3-noisy.pdf")
    plt.show()

    for filter_size in [2, 3, 5, 7, 11]:
        X_denoised = sp.ndimage.uniform_filter(X_noisy, size=filter_size)
        signal_to_noise_ratio = get_signal_to_noise_ratio(X, power_of_X, X_denoised)
        show_denoised_image(X_denoised, filter_size, signal_to_noise_ratio)


if __name__ == '__main__':
    main()
