import numpy as np
import matplotlib.pyplot as plt


def generate_time_series(N):
    discrete_time_interval = np.linspace(0, 10, N)
    trend = lambda t: 0.5 * t ** 2
    seasonal = lambda t: 2 * np.sin(2 * np.pi * 8 * t) + 1.5 * np.cos(2 * np.pi * 4 * t)
    residuals = np.random.normal(scale=2, size=N)
    time_series = trend(discrete_time_interval) + seasonal(discrete_time_interval) + residuals

    return trend, seasonal, residuals, time_series, discrete_time_interval


class AutoRegressionModel:
    def __init__(self, p, m):
        self.data = None
        self.N = None
        self.p = p
        self.m = m
        self.coefficients = None

    def fit(self, data):
        if self.data:
            print("AutoRegression model already fitted.")
            return

        if len(data) <= self.m + self.p:
            print("Not enough data points to fit the model.")
            return

        self.data = data
        self.N = len(data)

        # saving y and Y according to the definition of the model
        y = self.data[:self.N - 1 - self.m:-1]
        Y = np.ndarray((self.m, self.p))
        for i in range(self.m):
            Y[i] = self.data[self.N - i - 2:self.N - i - 2 - self.p:-1]

        self.coefficients = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(y))

    def predict(self):
        if self.data is None:
            print("AutoRegression model not fitted yet.")
            return

        prediction = self.coefficients.dot(self.data[:self.N - 1 - self.p:-1])
        return prediction

    def update(self, value):
        if self.data is None:
            print("AutoRegression model not fitted yet.")
            return

        self.data = np.append(self.data, value)
        self.N += 1


class MovingAverageModel:
    def __init__(self, q, m):
        self.data = None
        self.error = None
        self.N = None
        self.q = q
        self.m = m
        self.coefficients = None

    def fit(self, data):
        if self.data:
            print(f"MovingAverage model already fitted.")
            return

        if len(data) < self.q + self.m:
            print("Not enough data points to fit the model.")
            return

        self.data = data
        self.error = np.random.normal(size=len(data))
        self.N = len(data)

        # saving y and E according to the definition of the model
        y = self.data[:self.N - 1 - self.m:-1]
        E = np.ndarray((self.m, self.q))
        for i in range(self.m):
            E[i] = self.error[self.N - 2 - i:self.N - 2 - i - self.q:-1]

        self.coefficients = np.linalg.inv(E.T.dot(E)).dot(E.T.dot(y))

    def predict(self):
        if self.data is None:
            print("MovingAverage model not fitted yet.")
            return

        prediction = self.coefficients.dot(self.error[:self.N - 1 - self.q:-1])
        return prediction

    def update(self, value, error=None):
        if self.data is None:
            print("MovingAverage model not fitted yet.")
            return

        if not error:
            prediction = self.predict()
            self.error = np.append(self.error, value - prediction)
        else:
            self.error = np.append(self.error, error)

        self.data = np.append(self.data, value)
        self.N += 1


class AutoRegressiveMovingAverageModel:
    def __init__(self, p, q, m, n):
        self.auto_regression_model = AutoRegressionModel(p, m)
        self.moving_average_model = MovingAverageModel(q, n)

    def fit(self, data):
        self.auto_regression_model.fit(data)
        self.moving_average_model.fit(data)

    def predict(self):
        return self.auto_regression_model.predict() + self.moving_average_model.predict()

    def update(self, value):
        prediction = self.predict()
        self.auto_regression_model.update(value)
        self.moving_average_model.update(value, value - prediction)


def cross_validate(data, p, q, m, n, n_fold=10):
    number_of_elements_per_fold = int(len(data) * (1 / n_fold))

    if number_of_elements_per_fold <= p + m or number_of_elements_per_fold <= q + n:
        print("Not enough data points to perform cross validation with the desired number of folds.")
        return

    prediction_error = 0
    for fold_index in range(n_fold):
        train = data[fold_index * number_of_elements_per_fold:(fold_index + 1) * number_of_elements_per_fold - 1]
        test = data[(fold_index + 1) * number_of_elements_per_fold - 1]

        auto_regressive_moving_average_model = AutoRegressiveMovingAverageModel(p, q, m, n)
        auto_regressive_moving_average_model.fit(train)
        prediction = auto_regressive_moving_average_model.predict()
        prediction_error += (test - prediction) ** 2

    prediction_error /= n_fold
    return prediction_error


def perform_cross_validation(data, p, q, m, n, n_fold=10):
    best_p = -1
    best_q = -1
    best_m = -1
    best_n = -1
    best_score = np.inf

    for current_p in p:
        for current_q in q:
            for current_m in m:
                for current_n in n:
                    score = cross_validate(data, current_p, current_q, current_m, current_n, n_fold)
                    if score < best_score:
                        best_score = score
                        best_p = current_p
                        best_q = current_q
                        best_m = current_m
                        best_n = current_n

    return best_p, best_q, best_m, best_n, best_score


def main():
    trend, seasonal, residuals, time_series, discrete_time_interval = generate_time_series(1000)

    auto_regressive_moving_average_model = AutoRegressiveMovingAverageModel(23, 17, 200, 150)
    auto_regressive_moving_average_model.fit(time_series[:980])

    predictions = []
    for i in range(20):
        predictions.append(auto_regressive_moving_average_model.predict())
        auto_regressive_moving_average_model.update(time_series[980 + i])

    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Auto Regression Moving Average Model")
    figure.set_size_inches(25, 15)

    axis.plot(time_series, label="Time Series")
    axis.plot(np.arange(980, 1000), predictions, label="Predictions")
    axis.legend()

    plt.savefig("exercise-4.png")
    plt.savefig("exercise-4.pdf")
    plt.show()

    # performing cross validation to determine the optimal hyperparameters
    print("Performing cross validation. This may take a while...")
    optimal_p, optimal_q, optimal_m, optimal_n, score = perform_cross_validation(time_series, range(1, 21), range(1, 21), [25, 50, 75], [20, 50, 75])
    print(f"Obtained score={score:.2f} for p={optimal_p}, q={optimal_q}, m={optimal_m}, n={optimal_n}")


if __name__ == '__main__':
    main()
