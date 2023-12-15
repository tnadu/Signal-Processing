import numpy as np
import matplotlib.pyplot as plt


def generate_time_series(N):
    discrete_time_interval = np.linspace(0, 10, N)
    trend = lambda t: 0.5 * t ** 2
    seasonal = lambda t: 2 * np.sin(2 * np.pi * 8 * t) + 1.5 * np.cos(2 * np.pi * 4 * t)
    residuals = np.random.normal(scale=2, size=N)
    time_series = trend(discrete_time_interval) + seasonal(discrete_time_interval) + residuals

    return trend, seasonal, residuals, time_series, discrete_time_interval


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


def main():
    trend, seasonal, residuals, time_series, discrete_time_interval = generate_time_series(1000)

    moving_average_model = MovingAverageModel(11, 300)
    moving_average_model.fit(time_series[:980])

    predictions = []
    for i in range(20):
        predictions.append(moving_average_model.predict())
        moving_average_model.update(time_series[980 + i])

    figure, axis = plt.subplots(1, layout="constrained")
    figure.suptitle("Moving Average Model")
    figure.set_size_inches(25, 15)

    axis.plot(time_series, label="Time Series")
    axis.plot(np.arange(980, 1000), predictions, label="Predictions")
    axis.legend()

    plt.savefig("exercise-3.png")
    plt.savefig("exercise-3.pdf")
    plt.show()


if __name__ == '__main__':
    main()
