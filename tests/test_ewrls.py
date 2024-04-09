import time
import unittest

import numpy as np
import pandas as pd
# import statsmodels.api as sm
from statsmodels import api as sm
import sys
# from strat_macro_qrv.models.recursive_ls.rls.recursive_ols import RecursiveOLS
# from strat_macro_qrv.models.recursive_ls.rls.ewrls import EWRLSRidge
 #..src.utilities import LOGGER
sys.path.append('../src')
from utilities import LOGGER #as LOGGER



class MyTestCase(unittest.TestCase):

    def define_data(self, length=1500):
        self.length = length
        self.exog = np.append(np.ones((self.length, 1)), np.random.randn(self.length, 3), axis=1)
        self.betas = np.array([4, 0.5, 3, 1])
        self.sigma = 0.2
        self.y = self.exog @ self.betas[:, np.newaxis] + np.random.randn(self.length, 1) * self.sigma
        self.const = np.ones((self.length, 1))

    @staticmethod
    def exp_weights(vec_length, span):
        if span >= 1:
            alpha = 2 / (span + 1)
        else:
            LOGGER.error('Span must be >=1')

        T = vec_length
        wt_vec = {t: (1 - alpha) ** (T - t - 1) for t in range(T)}
        # make sure wt_vec = 1 for most recent obs
        weights = pd.DataFrame(wt_vec.values(), index=wt_vec.keys(), columns=['exp_weights'])
        return weights

    def test_ewrls(self):
        for length in [100, 500, 1500, 3000]:

            self.define_data(length=length)

            for ewm_span in [20, 30, 50, 100, 500, np.inf]:

                if ewm_span < np.inf:
                    wts = self.exp_weights(len(self.y), span=ewm_span)
                    # wts.index = self.index
                else:
                    wts = pd.DataFrame(np.ones((len(self.y), 1)))  # , index=self.y.index)
                    # i.e. the same as OLS

                # https: // www.statsmodels.org / stable / generated / statsmodels.regression.linear_model.WLS.fit_regularized.html
                # statsmodels.regression.linear_model.WLS.fit_regularized

                if ewm_span < np.inf:
                    wls = sm.WLS(self.y, self.exog, wts)  # ela   stic net here too

                else:
                    wls = sm.WLS(self.y, self.exog)  # ela   stic net here too
                fit = wls.fit()
                beta_wls = fit.params

                ewrls = EWRLSRidge(num_features=self.exog.shape[1], span=ewm_span, regularization=0.0)
                hot_start = True

                if hot_start:
                    # start is 10 in
                    cov = pd.DataFrame(self.exog[:11, :]).ewm(span=ewm_span).cov().loc[10]
                    # start with little piece of data

                    p = np.linalg.pinv(cov)
                    nobs = 11
                    mini_wts = self.exp_weights(11, span=ewm_span)
                    beta = sm.WLS(self.y[:11], self.exog[:11, :]).fit().params
                    # print(beta)
                else:
                    p = np.eye(self.exog.shape[1]) * self.sigma ** 2 * 101.5
                    beta = np.zeros((4, 1))
                    nobs = 0

                ewrls.set_start(p, beta, nobs)
                ewrls.update(self.y[nobs:], self.exog[nobs:, :])
                beta_ewrls = ewrls.beta[:, 0]

                for i in range(4):
                    # print('assert  {}'.format(i))
                    # print(' beta  = {}'.format(beta_ewrls[i]) + ' beta_wls[i] = {}'.format(beta_wls[i]))
                    # very roughly equal.
                    self.assertAlmostEqual(beta_ewrls[i], beta_wls[i], places=0)

    def test_expwtls(self):
        for length in [100, 500, 1500, 3000]:
            self.define_data(length=length)

            for ewm_span in [20, 30, 50, 100, 500, np.inf]:

                y_frame = pd.DataFrame(self.y)

                if ewm_span < np.inf:
                    movav = y_frame.ewm(span=ewm_span).mean()
                    end_val = movav.iloc[-1, 0]
                else:
                    end_val = y_frame.mean().iloc[0]

                if ewm_span < np.inf:
                    wts = self.exp_weights(len(self.y), span=ewm_span)
                    # wts.index = self.y.index
                else:
                    wts = pd.DataFrame(np.ones((len(self.y), 1)))
                    # i.e. the same as OLS

                if ewm_span < np.inf:
                    wls2 = sm.WLS(self.y, self.const, wts)  # ela   stic net here too

                else:
                    wls2 = sm.WLS(self.y, self.const)  # ela   stic net here too
                fit = wls2.fit()
                ewm_beta = fit.params

                self.assertAlmostEqual(ewm_beta[0], end_val, places=5)  # , message=None, verbose=True)


if __name__ == '__main__':
    unittest.main()


def bench_ols_rls2(y, X):
    '''
    Benchmarking Iterative Rolling OLS
    '''

    length = X.shape[0]
    k_dim = X.shape[1]

    yhat_roll = pd.DataFrame(np.zeros((length, 1)))
    betahat_roll = pd.DataFrame(np.zeros((length, k_dim)))
    sigma_data = [[n, np.zeros((k_dim, k_dim))] for n in range(length)]
    sigma2hat_roll = pd.DataFrame(sigma_data).drop(0, axis=1)

    t0_2 = time.time()
    for t in np.arange(50, length - 1):

        features = X.iloc[t - 50:t + 1, :]
        mod = sm.OLS(y.iloc[t - 50:t + 1, :], features)
        fitted = mod.fit()

        xslice = X.iloc[t + 1, :]
        beta_h = fitted.params
        temp = 0
        for k in range(k_dim):
            temp += beta_h[k] * xslice[k]
        yhat_roll.iloc[t + 1, :] = temp
        # betahat.iloc[t+1,:].dot( X.iloc[t+1,:].T)
        betahat_roll.iloc[t + 1, :] = beta_h
        # x[t+1]  #fitted.predict(x[t+1])

    t1_2 = time.time()
    print('Python Rolling Interative OLS: {}'.format(t1_2 - t0_2))  # 58 secs

    '''
    Comparing to Recursive Rolling LS
    '''
    yhat_roll1 = pd.DataFrame(np.zeros((length, 1)))
    betahat_roll1 = pd.DataFrame(np.zeros((length, k_dim)))
    sigma_data = [[n, np.zeros((k_dim, k_dim))] for n in range(length)]
    sigma2hat_roll1 = pd.DataFrame(sigma_data).drop(0, axis=1)

    t0_3 = time.time()

    # initate - stupid if else: doesn't work below!
    rls_instance1 = RecursiveOLS(y.iloc[:51, :].values,
                                 X.iloc[:51, :].values)
    beta_h = rls_instance1.beta

    betahat_roll1.iloc[51, :] = beta_h.T
    # sigma2hat1.iloc[t+1,0] = p
    # create forecast and make scalar
    x_new = X.iloc[51, :]  # need _slice2col (recast type) if using matrix mult
    # yhat1[t+1] = (beta.T @ x_new)[0,0]
    temp = 0
    for i in range(k_dim):  # beta is col vec
        temp += beta_h[i][0] * x_new[i]
    yhat_roll1.iloc[51, 0] = temp

    for t in np.arange(51, length - 1):
        rls_instance1.rolling_add(y.iloc[t, :].values,
                                  X.iloc[t, :].values)
        beta_h = rls_instance1.beta
        # _run_update only with last values

        betahat_roll1.iloc[t + 1, :] = beta_h.T
        # sigma2hat1.iloc[t+1,0] = p
        # create forecast and make scalar
        x_new = X.iloc[t + 1, :]  # need _slice2col (recast type) if using matrix mult
        # yhat1[t+1] = (beta.T @ x_new)[0,0]
        temp = 0
        for i in range(k_dim):  # beta is col vec
            temp += beta_h[i][0] * x_new[i]
        yhat_roll1.iloc[t + 1, 0] = temp

    t1_3 = time.time()
    print('Python Rolling LS: {}'.format(t1_3 - t0_3))  # 58 secs

    # assert ((betahat_roll - betahat_roll1) < 10E-10).all().all()
