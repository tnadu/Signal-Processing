import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def generate_time_series(N):
    discrete_time_interval = np.linspace(0, 10, N)
    trend = lambda t: 0.5 * t ** 2
    seasonal = lambda t: 2 * np.sin(2 * np.pi * 8 * t) + 1.5 * np.cos(2 * np.pi * 4 * t)
    residuals = np.random.normal(scale=2, size=N)
    time_series = trend(discrete_time_interval) + seasonal(discrete_time_interval) + residuals

    return trend, seasonal, residuals, time_series, discrete_time_interval


def get_autocorrelation_vector(vector, number_of_correlated_elements):
    N = len(vector)
    autocorrelation_vector = np.ndarray((N - number_of_correlated_elements + 1))

    for lag in range(N - number_of_correlated_elements + 1):
        autocorrelation_vector[lag] = np.dot(vector[N - number_of_correlated_elements:], vector[N - number_of_correlated_elements - lag:N - lag])

    return autocorrelation_vector


def cross_validate(data, p, m, n_fold=10):
    number_of_elements_per_fold = int(len(data) * (1 / n_fold))

    if number_of_elements_per_fold <= p + m:
        print("Not enough data points to perform cross validation with the desired number of folds.")
        return

    prediction_error = 0
    for fold_index in range(n_fold):
        train = data[fold_index * number_of_elements_per_fold:(fold_index + 1) * number_of_elements_per_fold - 1]
        test = data[(fold_index + 1) * number_of_elements_per_fold - 1]

        autoregression_model = AutoRegressionModel()
        autoregression_model.fit(train, p, m)
        prediction = autoregression_model.predict(train, 1)
        prediction_error += (test - prediction) ** 2

    prediction_error /= n_fold
    return prediction_error


def perform_cross_validation(data, p, m, n_fold):
    best_p = -1
    best_m = -1
    best_score = np.inf

    for current_p in p:
        for current_m in m:
            score = cross_validate(data, current_p, current_m, n_fold)
            if score < best_score:
                best_score = score
                best_p = current_p
                best_m = current_m

    return best_p, best_m, best_score[0]


class AutoRegressionModel:
    def __init__(self):
        self.data = None
        self.N = None
        self.p = None
        self.m = None
        self.coefficients = None

    def fit(self, data, p, m, use_yule_walker_equations=False):
        if self.data or self.p or self.m:
            print("AutoRegression model already fitted.")
            return

        if len(data) <= m + p:
            print("Not enough data points to fit the model.")
            return

        self.data = data
        self.N = len(data)
        self.p = p
        self.m = m

        if not use_yule_walker_equations:
            # saving y and Y according to the definition of the model
            y = self.data[:self.N - self.m - 1:-1]
            Y = np.ndarray((self.m, self.p))
            for i in range(self.m):
                Y[i] = self.data[self.N - i - 2:self.N - i - 2 - self.p:-1]

            self.coefficients = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(y))
        else:
            autocorrelation_vector = get_autocorrelation_vector(self.data[self.N - self.m - self.p:], self.m)
            autocorrelation_matrix = sp.linalg.toeplitz(autocorrelation_vector[:self.p])
            self.coefficients = np.linalg.inv(autocorrelation_matrix).dot(autocorrelation_vector[1:])

    def predict(self, data, number_of_predictions):
        if self.data is None or not self.p or not self.m:
            print("AutoRegression model not fitted yet.")
            return

        N = len(data)
        predictions = np.ndarray(number_of_predictions)

        for i in range(number_of_predictions):
            # using as many previous predictions as possible,
            # and filling any missing values using the dataset
            current_data_horizon = np.concatenate((data[N - self.p + i:], predictions[max(0, i - self.p):i]))
            predictions[i] = current_data_horizon.dot(self.coefficients)

        return predictions


def main():
    trend, seasonal, residuals, time_series, discrete_time_interval = generate_time_series(1000)

    # a)
    figure, axes = plt.subplots(4, layout="constrained")
    figure.set_size_inches(25, 15)

    axes[0].plot(discrete_time_interval, trend(discrete_time_interval))
    axes[0].set_title("Trend")

    axes[1].plot(discrete_time_interval, seasonal(discrete_time_interval))
    axes[1].set_title("Seasonal")

    axes[2].plot(discrete_time_interval, residuals)
    axes[2].set_title("Residuals")

    axes[3].plot(discrete_time_interval, time_series, label="")
    axes[3].set_title("Time Series")

    plt.savefig("exercise-1-figure-1.png")
    plt.savefig("exercise-1-figure-1.pdf")
    plt.show()

    # b)
    autocorrelation_vector = get_autocorrelation_vector(time_series, 500)
    autocorrelation_vector_numpy = np.correlate(time_series, time_series[500:], mode="valid")
    # match the element order of the custom function implementation
    autocorrelation_vector_numpy = np.flip(autocorrelation_vector_numpy)
    print(f"b) Are the two autocorrelation vectors all-close?: {np.allclose(autocorrelation_vector, autocorrelation_vector_numpy)}")

    figure, axis = plt.subplots(1, layout="constrained")
    figure.set_size_inches(15, 10)
    figure.suptitle("Autocorrelation of the time series, size 500")

    axis.plot(autocorrelation_vector)
    axis.set_xlabel("Lag")

    plt.savefig("exercise-1-figure-2.png")
    plt.savefig("exercise-1-figure-2.pdf")
    plt.show()

    # c)
    autoregression_model = AutoRegressionModel()
    autoregression_model.fit(time_series[:900], 23, 100)
    predictions = autoregression_model.predict(time_series[:900], 100)

    figure, axis = plt.subplots(1, layout="constrained")
    figure.set_size_inches(25, 15)
    figure.suptitle("Autoregression model predictions")

    axis.plot(time_series, label="Actual")
    axis.plot(np.array(range(900, 1000)), predictions, label="Predictions")
    axis.legend()

    plt.savefig("exercise-1-figure-3.png")
    plt.savefig("exercise-1-figure-3.pdf")
    plt.show()

    # d)
    p, m, score = perform_cross_validation(time_series, [7, 11, 13, 17, 19, 23], [30, 50, 70], 10)
    print(f"Obtained score={score:.2f} for p={p}, m={m}")


if __name__ == '__main__':
    main()
