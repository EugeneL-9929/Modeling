import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# pip install statsmodels
# AR model
from statsmodels.tsa.ar_model import AutoReg
# ARIMA model
from statsmodels.tsa.arima.model import ARIMA


def dataPreparations():
    np.random.seed(0)
    data = np.random.randn(100)
    timeSeries = pd.Series(data)
    return timeSeries


def autoRegressions(timeSeries):
    model = AutoReg(timeSeries, lags=1)
    modelFit = model.fit()
    print(modelFit.summary())
    return modelFit

def autoRegressionMovingAverages(timeSeries):
    # AR Difference MA
    model = ARIMA(timeSeries, order=(0, 0, 1))
    modelFit = model.fit()
    print(modelFit.summary())
    return modelFit

def main():
    timeSeries = dataPreparations()
    model = autoRegressionMovingAverages(timeSeries)
    predictions = model.forecast(steps=6)
    # predictions = model.predict(start=len(timeSeries), end=len(timeSeries) + 5)
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(timeSeries)), timeSeries, label='Original Data')
    plt.scatter(range(len(timeSeries), len(timeSeries) + 6), predictions, label='Predictions', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
