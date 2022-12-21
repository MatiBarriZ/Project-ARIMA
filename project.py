import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy as sp
import math
from scipy.stats import skew, kurtosis, chi2, shapiro, normaltest, anderson, chisquare, kstest
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def perform_test_adfuller(series, alpha = 0.05):
    result = adfuller(series)
    x_is_stationary = (result[1] > alpha)
    # print('ADF Statistic: %f' %result[0])
    # print('p-value: %f' %result[1])
    return x_is_stationary

# If the p-value ≤ 0.05, then we reject the null hypothesis i.e. we assume the distribution of our variable is not normal/gaussian
# If the p-value > 0.05, then we fail to reject the null hypothesis i.e. we assume the distribution of our variable is normal/gaussian.

def jarque_bera(x, alpha = 0.05):
    nb_sims = 10**6
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_skew = skew(x)
    x_kurtosis = kurtosis(x) # excess kurtosis
    x_jb_stat = nb_sims/6*(x_skew**2 + 1/4*x_kurtosis**2)
    x_p_value = 1 - chi2.cdf(x_jb_stat, df=2)
    x_is_normal = (x_p_value > alpha) # equivalently jb < 6 if alpha = 0.05

    return x_is_normal
    # print('mean is ' + str(x_mean))
    # print('standard deviation is ' + str(x_std))
    # print('skewness is ' + str(x_skew))
    # print('kurtosis is ' + str(x_kurtosis))
    # print('JB statistic is ' + str(x_jb_stat))
    # print('p-value ' + str(x_p_value))
    # print('is normal ' + str(x_is_normal))

def shapiro_wilk(x, alpha = 0.05):
    stat, p = shapiro(x)
    x_is_normal = (p > alpha)
    print('W statistic is ' + str(stat))
    print('p-value ' + str(p))
    print('is normal ' + str(x_is_normal))

def dangostino_k_sqrt(x, alpha = 0.05):
    stat, p = normaltest (x)
    x_is_normal = (p > alpha)
    print('K statistic is ' + str(stat))
    print('p-value ' + str(p))
    print('is normal ' + str(x_is_normal))

def anderson_darling(x, alpha = 0.05):
    result = anderson(x)
    print('A statistic is ' + str(result.statistic))
    for i in range(len(result.critical_values)):
        sig_lev, crit_val = result.significance_level[i], result.critical_values[i]
        x_is_normal = (result.statistic < crit_val)
        if sig_lev == alpha*100:
            print('is normal ' + str(x_is_normal))

def chi_square(x, alpha=0.05):
    statistic,pvalue = chisquare(x)
    print('statistic=%.3f, p=%.3f\n' % (statistic, pvalue))
    x_is_normal = (pvalue > alpha)
    print('is normal ' + str(x_is_normal))

def lilliefors_test(x, alpha=0.05):
    statistic,pvalue = lilliefors(x)
    print('statistic=%.3f, p=%.3f\n' % (statistic, pvalue))
    x_is_normal = (pvalue > alpha)
    print('is normal ' + str(x_is_normal))

def kolmogorov_smirnov(x, alpha = 0.05):
    statistic, pvalue = kstest(x, 'norm')
    print('statistic=%.3f, p=%.3f\n' % (statistic, pvalue))
    x_is_normal = (pvalue > alpha)
    print('is normal ' + str(x_is_normal))

def least_squares(x, y):
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    return np.linalg.inv((x.T @ x)) @ (x.T @ y)

def difference(x, d = 1, prt = False):
    # https://numpy.org/doc/stable/reference/generated/numpy.r_.html
    if d == 0:
        return x
    else:
        x = np.r_[x[0], np.diff(x)]
        if prt:
            print(x, d)
        return difference(x, d - 1)

def undo_difference(x, d = 1, prt = False): # Entender bien para que se utiliza
    # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html?highlight=cumsum#numpy.cumsum
    if d == 1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        if prt:
            print(x, d)
        return undo_difference(x, d - 1)

def lag_view(x, order):
    """
    For every value X_i create a row that lags k values: [X_i-1, X_i-2, ... X_i-k]
    """
    y = x.copy()
    # Create features by shifting the window of `order` size by one step.
    # This results in a 2D array [[t1, t2, t3], [t2, t3, t4], ... [t_k-2, t_k-1, t_k]]
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])

    # Reverse the array as we started at the end and remove duplicates.
    # Note that we truncate the features [order -1:] and the labels [order]
    # This is the shifting of the features with one time step compared to the labels
    # https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:]

    return x, y

def pearson_correlation(x, y):
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()) #lo entiendo

def acf(x, lag=40): #lo entiendo
    """
    Determine autocorrelation factors.
    :param x: (array) Time series.
    :param lag: (int) Number of lags.
    """
    return np.array([1] + [pearson_correlation(x[:-i], x[i:]) for i in range(1, lag)])

def bartletts_formula(acf_array, n):#lo entiendo
    """
    Computes the Standard Error of an acf with Bartlet's formula
    Read more at: https://en.wikipedia.org/wiki/Correlogram
    :param acf_array: (array) Containing autocorrelation factors
    :param n: (int) Length of original time series sequence.
    """
    # The first value has autocorrelation with it self. So that values is skipped
    se = np.zeros(len(acf_array) - 1)
    se[0] = 1 / np.sqrt(n)
    se[1:] = np.sqrt((1 + 2 * np.cumsum(acf_array[1:-1]**2)) / n ) # 1-2 o 1+2?
    return se

def plot_acf(x, alpha=0.05, lag=40):#lo entiendo
    """
    :param x: (array)
    :param alpha: (flt) Statistical significance for confidence interval.
    :parm lag: (int)
    """
    acf_val = acf(x, lag)
    plt.figure(figsize=(16, 4))
    plt.vlines(np.arange(lag), 0, acf_val)
    plt.scatter(np.arange(lag), acf_val, marker='o')
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')

    # Determine confidence interval
    ci = sp.stats.norm.ppf(1 - alpha / 2.) * bartletts_formula(acf_val, len(x))
    y = [0.0 for i in range(lag+2)]
    x_lag = [-1]
    for i in range(lag+1):
        x_lag.append(i)
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)
    plt.plot(x_lag, y, '-')
    plt.show()

def pacf(x, lag=20): # An other option is https://github.com/RJTK/Levinson-Durbin-Recursion
    """
    Partial autocorrelation function.
    
    https://en.wikipedia.org/wiki/Partial_autocorrelation_function

    pacf results in:
        [1, acf_lag_1, pacf_lag_2, pacf_lag_3]
    :param x: (array)
    :param lag: (int)
    """
    y = []

    # Partial auto correlation needs intermediate terms.
    # Therefore we start at index 3
    for i in range(3, lag + 2):
        backshifted = lag_view(x, i)[0]

        xt = backshifted[:, 0]
        feat = backshifted[:, 1:-1]
        xt_hat = LinearModel(fit_intercept=False).fit_predict(feat, xt)

        xt_k = backshifted[:, -1]
        xt_k_hat = LinearModel(fit_intercept=False).fit_predict(feat, xt_k)

        y.append(pearson_correlation(xt - xt_hat, xt_k - xt_k_hat))
    return np.array([1, acf(x, 2)[1]] +  y)

def plot_pacf(x, alpha=0.05, lag=40, title=None):
    """
    :param x: (array)
    :param alpha: (flt) Statistical significance for confidence interval.
    :parm lag: (int)
    """
    pacf_val = pacf(x, lag)
    plt.figure(figsize=(16, 4))
    plt.vlines(np.arange(lag + 1), 0, pacf_val)
    plt.scatter(np.arange(lag + 1), pacf_val, marker='o')
    plt.axhline(0, color="black")
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')

    # Determine confidence interval
    ci =sp.stats.norm.ppf(1 - alpha / 2.) * bartletts_formula(pacf_val, len(x))
    y = [0.0 for i in range(lag+2)]
    x_lag = [-1]
    for i in range(lag+1):
        x_lag.append(i)
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)
    plt.plot(x_lag, y, '-')
    plt.show()

def plot_serie(stars, star, savefig, show, typetime = 'mjd'):
    plt.figure(figsize = (40,10))
    plt.plot(stars[star][typetime], stars[star]['mag'], 'o')
    plt.ylabel('magnitud')
    plt.title(star)
    plt.tick_params(labelrotation=45)
    plt.text.usetex = True
    if savefig:
        plt.savefig(format = 'svg')
    if show:
        plt.show()

def plot_forecast(X, y, X_test, y_test, y_pred, X_train, y_train, title, savefig, show, typetime = 'mjd'):
    plt.figure(figsize = (40,10))
    plt.plot(X, y, '.-', color = 'black', label = 'original') #.-
    plt.plot(X_train, y_train, '.-', color = 'red', label = 'train') #.-
    # plt.plot(X_test, y_pred, '.-', color = 'blue', label = 'forecast_test') #.-
    plt.legend(loc='best')
    plt.ylabel('magnitud')
    plt.xlabel('time [mjd]')
    plt.title(str(title))
    plt.tick_params(labelrotation=45)
    plt.text.usetex = True
    plt.show()

def qqplot(data):
    m = data.mean()
    st = data.std()

    # Standardize the data
    for i in range(0,data.shape[0],1):
        data.iloc[i] = (data.iloc[i]-m)/st

    # determine standard quantiles from the standard normal data

    mu, sigma = 0, 1 # mean and standard deviation
    s = np.random.normal(mu, sigma, len(data))
    q = []
    j=0
    for i in range(1,len(s)+1,1):
        j=i/data.shape[0]
        q_temp = np.quantile(s, j)
        q.append(q_temp)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(q,sorted(data),'o')
    plt.plot(s,s, '-')
    plt.xlabel("Quantile of standard normal distribution")
    plt.ylabel("Sample Z-score")
    plt.title('Q-Q Plot')
    plt.grid()
    plt.show()

def ma_process(eps, theta):
    """
    Creates an MA(q) process with a zero mean (mean not included in implementation).
    :param eps: (array) White noise signal.
    :param theta: (array/ list) Parameters of the process.
    """
    # reverse the order of theta as Xt, Xt-1, Xt-k in an array is Xt-k, Xt-1, Xt.
    theta = np.array([1] + list(theta))[::-1][:, None]
    eps_q, _ = lag_view(eps, len(theta))
    return eps_q @ theta

def ar_process(eps, phi):
    """
    Creates a AR process with a zero mean.
    """
    # Reverse the order of phi and add a 1 for current eps_t
    phi = np.r_[1, phi][::-1]
    ar = eps.copy()
    offset = len(phi)
    for i in range(offset, ar.shape[0]):
        ar[i - 1] = ar[i - offset: i] @ phi
    return ar

def mean_absolute_percentage_error(y_true, y_pred):
    # y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def mean_squared_error(y_true, y_pred):
    mse = np.square(np.subtract(y_true,y_pred)).mean()
    return mse

def root_mean_squared_error(y_true, y_pred):
    mse = np.square(np.subtract(y_true,y_pred)).mean()
    rmse = math.sqrt(mse)
    return rmse

class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept #booleano
        self.beta = None
        self.intercept_ = None
        self.coef_ = None

    def _prepare_features(self, x): # coloca los 1 que son necesarios para obtener el intercepto.
        # https://numpy.org/doc/stable/reference/generated/numpy.shape.html
        # https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y): #Conjunto de entrenamiento y test?
        x = self._prepare_features(x)
        self.beta = least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta

    # no entiendo la utilidad de fit_predict
    def predict(self, x):
        x = self._prepare_features(x)
        return x @ self.beta

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)

class ARIMA(LinearModel):
    def __init__(self, q, d, p):
        """
        An ARIMA model.
        :param q: (int) Order of the MA model.
        :param p: (int) Order of the AR model.
        :param d: (int) Number of times the data needs to be differenced.
        """
        super().__init__(True) # Herencia de la superclass LinearModel a la subclass ARIMA.
        self.p = p
        self.d = d
        self.q = q
        self.ar = None
        self.resid = None

    def prepare_features(self, x):
        if self.d > 0:
            x = difference(x, self.d)

        ar_features = None
        ma_features = None

        # Determine the features and the epsilon terms for the MA process
        if self.q > 0:
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p)
                self.ar.fit_predict(x)
            eps = self.ar.resid
            eps[0] = 0

            # prepend with zeros as there are no residuals_t-k in the first X_t
            ma_features, _ = lag_view(np.r_[np.zeros(self.q), eps], self.q)

        # Determine the features for the AR process
        if self.p > 0:
            # prepend with zeros as there are no X_t-k in the first X_t
            ar_features = lag_view(np.r_[np.zeros(self.p), x], self.p)[0]

        if ar_features is not None and ma_features is not None:
            n = min(len(ar_features), len(ma_features))
            ar_features = ar_features[:n]
            ma_features = ma_features[:n]
            features = np.hstack((ar_features, ma_features))
        elif ma_features is not None:
            n = len(ma_features)
            features = ma_features[:n]
        else:
            n = len(ar_features)
            features = ar_features[:n]

        return features, x[:n]

    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x)
        return features

    def fit_predict(self, x):
        """
        Fit and transform input
        :param x: (array) with time series.
        """
        features = self.fit(x)
        return self.predict(x, prepared=(features))

    def predict(self, x, **kwargs):
        """
        :param x: (array)
        :kwargs:
            prepared: (tpl) containing the features, eps and x
        """
        features = kwargs.get('prepared', None)
        if features is None:
            features, x = self.prepare_features(x)

        y = super().predict(features)
        self.resid = x - y

        return self.return_output(y)

    def return_output(self, x):
        if self.d > 0:
            x = undo_difference(x, self.d)
        return x

    def forecast(self, x, n):
        """
        Forecast the time series.

        :param x: (array) Current time steps.
        :param n: (int) Number of time steps in the future.
        """
        features, x = self.prepare_features(x)
        y = super().predict(features)

        # Append n time steps as zeros. Because the epsilon terms are unknown
        y = np.r_[y, np.zeros(n)]
        for i in range(n):
            feat = np.r_[y[-(self.p + n) + i: -n + i], np.zeros(self.q)]
            y[x.shape[0] + i] = super().predict(feat[None, :])
        return self.return_output(y)

def main():

    data = sm.datasets.sunspots.load_pandas().data
    x = data['SUNACTIVITY'].values.squeeze()
    plot_acf(x, lag=20)
    plot_pacf(x, lag=20)
    q = 1
    d = 0
    p = 3
    
    m = ARIMA(q, d, p)
    print(type(m))
    pred = m.fit_predict(x)
    print(pred)
    

if __name__ == '__main__':
    main()