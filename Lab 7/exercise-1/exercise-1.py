import numpy as np
import matplotlib.pyplot as plt


def get_fft_first_version(N):
    center = N // 2
    Y = np.zeros((N, N))
    Y[center-5][center] = 1
    Y[center+5][center] = 1
    return Y


def get_fft_second_version(N):
    center = N // 2
    Y = np.zeros((N, N))
    Y[center][center+5] = 1
    Y[center][center-5] = 1
    return Y


def get_fft_third_version(N):
    center = N // 2
    Y = np.zeros((N, N))
    Y[center-5][center-5] = 1
    Y[center+5][center+5] = 1
    return Y


def main():
    # a)
    # frequency of y-axis component is 1.5, which explains why
    # the discernible rectangles are 1.5 times longer in the
    # y-axis than in the x-axis
    f = lambda x, y: np.sin(2 * np.pi * x + 3 * np.pi * y)
    X = np.fromfunction(f, (512, 512))
    plt.figure(figsize=(25, 14))
    plt.imshow(X, cmap=plt.cm.gray)
    plt.suptitle("X; f(x,y) = sin(2 * pi * x + 3 * pi * y)")
    plt.savefig("exercise-1-image-1.png")
    plt.savefig("exercise-1-image-1.pdf")
    plt.show()

    Y = np.fft.fft2(X)
    Y_db = 20 * np.log10(abs(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(Y_db)
    plt.colorbar()
    plt.suptitle("Y; f(x,y) = sin(2 * pi * x + 3 * pi * y)")
    plt.savefig("exercise-1-fft-1.png")
    plt.savefig("exercise-1-fft-1.pdf")
    plt.show()

    # b)
    f = lambda x, y: np.sin(4 * np.pi * x) + np.cos(6 * np.pi * y)
    X = np.fromfunction(f, (64, 64))
    plt.figure(figsize=(25, 14))
    plt.imshow(X, cmap=plt.cm.gray)
    plt.suptitle("X; f(x,y) = sin(4 * pi * x) + cos(6 * pi * y)")
    plt.savefig("exercise-1-image-2.png")
    plt.savefig("exercise-1-image-2.pdf")
    plt.show()

    Y = np.fft.fft2(X)
    Y_db = 20 * np.log10(abs(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(Y_db)
    plt.colorbar()
    plt.suptitle("Y; f(x,y) = sin(4 * pi * x) + cos(6 * pi * y)")
    plt.savefig("exercise-1-fft-2.png")
    plt.savefig("exercise-1-fft-2.pdf")
    plt.show()

    # c)
    Y = get_fft_first_version(513)
    Y_db = 20 * np.log10(abs(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(Y_db)
    plt.colorbar()
    plt.suptitle("Y; Y(0,5) = Y(0,N-5) = 1, else Y(m1,m2) = 0")
    plt.savefig("exercise-1-fft-3.png")
    plt.savefig("exercise-1-fft-3.pdf")
    plt.show()

    # ignore floating point rounding errors
    X = np.real(np.fft.ifft2(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(X, cmap=plt.cm.gray)
    plt.suptitle("X; Y(0,5) = Y(0,N-5) = 1, else Y(m1,m2) = 0")
    plt.savefig("exercise-1-image-3.png")
    plt.savefig("exercise-1-image-3.pdf")
    plt.show()

    # d)
    Y = get_fft_second_version(513)
    Y_db = 20 * np.log10(abs(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(Y_db)
    plt.colorbar()
    plt.suptitle("Y; Y(5,0) = Y(N-5,0) = 1, else Y(m1,m2) = 0")
    plt.savefig("exercise-1-fft-4.png")
    plt.savefig("exercise-1-fft-4.pdf")
    plt.show()

    X = np.real(np.fft.ifft2(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(X, cmap=plt.cm.gray)
    plt.suptitle("X; Y(5,0) = Y(N-5,0) = 1, else Y(m1,m2) = 0")
    plt.savefig("exercise-1-image-4.png")
    plt.savefig("exercise-1-image-4.pdf")
    plt.show()

    # e)
    Y = get_fft_third_version(513)
    Y_db = 20 * np.log10(abs(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(Y_db)
    plt.colorbar()
    plt.suptitle("Y; Y(5,5) = Y(N-5,N-5) = 1, else Y(m1,m2) = 0")
    plt.savefig("exercise-1-fft-5.png")
    plt.savefig("exercise-1-fft-5.pdf")
    plt.show()

    X = np.real(np.fft.ifft2(Y))
    plt.figure(figsize=(25, 14))
    plt.imshow(X, cmap=plt.cm.gray)
    plt.suptitle("X; Y(5,5) = Y(N-5,N-5) = 1, else Y(m1,m2) = 0")
    plt.savefig("exercise-1-image-5.png")
    plt.savefig("exercise-1-image-5.pdf")
    plt.show()


if __name__ == '__main__':
    main()
