import time

import numpy as np
import pandas as pd
from statsmodels import api as sm


class RecursiveOLS:
    '''
     RecursiveOLS procedures.

    Methods for fast _run_update adding a new observation and deleting an existing
    observation. Handles Increasing window and rolling window OLS.

    '''

    def __init__(self, y: np.ndarray, x: np.ndarray):
        '''
        recursivels_init [summary]
        [extended_summary]
        :param y: [description]
        :type y: pd..DataFrame
        :param x: [description]
        :type x: pd.DataFrame
        :return: [description]
        :rtype: [type]
        '''
        # Can we speed up by using QR? x = QR x'x = R'(Q'Q)R = R'R
        self.exog = x
        self.endog = y
        self.k_dim = x.shape[1]
        self.nobs = x.shape[0]
        # pinv(x'x) = pinv(R'R)
        prod = x.T.dot(x).astype(float)
        # # cast scalars as 1x1 arrays
        # if prod.shape == ():
        #     prod = np.array([[prod]])
        self.p = np.linalg.pinv(prod)
        self.beta = self.p.dot(x.T.dot(y)).astype(float)

    def update(self, y_update: np.ndarray,
               x_update: np.ndarray):
        '''
        y_t is 1-series of dependent data (1 obs)
        x_t is k-series of independent features (1 obs)
        beta is previous obs beta np.array (1,k) - col vec
        p is previous obs X'X np.array (k,k)
        don't transform x_t and y_t but make sure
        beta and p are correct dimensions at the end
        y_t is (1,) and x_t is (k,)
        '''

        '''
        Gain quantities
        '''
        p = self.p
        beta = self.beta
        k_dim = self.k_dim

        # 1/(1+h_{t+1,t+1})  where h is leverage
        c_update = 1 / (1 + x_update.T @ p @ x_update)
        k = (p @ x_update) * c_update
        # (num_features,)  vector
        # x_t is a (num_features,) vector, so cannot .T it
        self.p = ((np.eye(k_dim) - np.kron(k, x_update) \
                   .reshape([k_dim, k_dim])) @ p).astype(float)
        #    p - p.dot(x_t) * c * x_t.T.dot(P)
        self.beta = ((beta.T + k * (y_update \
                                    - x_update.T @ beta)[0]).T).astype(float)
        # kx1 vector
        self.nobs += 1
        self.exog = np.append(self.exog, x_update.reshape((1, k_dim)), axis=0)
        # print(y_t)
        # print(y_t.reshape((1,1)).shape)
        # print(self.endog.shape)
        self.endog = np.append(self.endog, y_update.reshape((1, 1)), axis=0)

    def delete(self, y_remove: np.ndarray,
               x_remove: np.ndarray):
        '''
        delete [summary]

        [extended_summary]

        :param y_remove: [description]
        :type y_remove: np.ndarray
        :param x_remove: [description]
        :type x_remove: np.ndarray
        '''
        p = self.p
        k_dim = self.k_dim
        beta = self.beta
        exog = self.exog
        endog = self.endog

        total_data = np.concatenate([endog, exog], axis=1)
        observation = np.concatenate([y_remove, x_remove], axis=0)

        find_obs = np.where(observation == total_data)
        # returns a tuple of arrays. If one has zero dim then emptly
        if find_obs[0].shape[0] == 0:
            raise ('Data cannot be deleted - was never part of original dataset')
        # remove the row with the instance
        total_data = np.delete(total_data, find_obs[0][0], 0)
        self.endog = total_data[:, 0].reshape((self.nobs - 1, 1))  # col vec
        self.exog = total_data[:, 1:]

        # c = 1/(1-h_tt)
        c_delete = 1 / (1 - x_remove.T @ p @ x_remove)
        k = (p @ x_remove) * c_delete
        self.p = ((np.eye(k_dim) + np.kron(k, x_remove) \
                   .reshape([k_dim, k_dim])) @ p).astype(float)
        self.beta = ((beta.T - k * (y_remove \
                                    - x_remove.T @ beta)[0]).T).astype(float)
        self.nobs -= 1

    def rolling_add(self, y_update: np.ndarray,
                    x_update: np.ndarray, roll_length=np.nan):
        '''
        rolling_add [summary]

        Do a rolling regression. If roll_lenght=np.nan continue with
        initialized dataaset length, pushing to end and popping from
        beginning of data. We should implement an increasing window version
        for start of dataset (i.e., min_sample_size). This should be
        basis for rolling

        :param y_update: [description]
        :type y_update: np.ndarray
        :param x_update: [description]
        :type x_update: np.ndarray
        :param roll_length: [description], defaults to np.nan
        :type roll_length: [type], optional
        '''
        if roll_length < self.k_dim:
            raise ('Cannot have too short a roll-window')
        self.update(y_update, x_update)
        if np.isnan(roll_length):
            x_remove = self.exog[0, :]
            y_remove = self.endog[0, :]
        else:
            x_remove = self.exog[-roll_length, :]
            y_remove = self.endog[-roll_length, :]
        self.delete(y_remove, x_remove)


def bench_ols_rls1(y, X):
    '''
    Benchmark iterating OLS
    '''
    length = X.shape[0]
    k_dim = X.shape[1]

    yhat = pd.DataFrame(np.zeros((length, 1)))
    betahat = pd.DataFrame(np.zeros((length, k_dim)))
    t0 = time.time()
    for t in np.arange(100, length - 1):
        features = X.iloc[:t + 1, :]
        mod = sm.OLS(y.iloc[:t + 1, :], features)
        fitted = mod.fit()
        temp = 0
        xslice = X.iloc[t + 1, :]
        beta_h = fitted.params
        for k in range(k_dim):
            temp += beta_h[k] * xslice[k]
        yhat.iloc[t + 1, :] = temp  # betahat.iloc[t+1,:].dot( X.iloc[t+1,:].T)
        betahat.iloc[t + 1, :] = beta_h
        # x[t+1]  #fitted.predict(x[t+1])
        t1 = time.time()
        print('Iterated OLS: {}'.format(t1 - t0))  # 21 secs

    '''
    Comparing Recursive LS
    '''

    yhat1 = pd.DataFrame(np.zeros((length, 1)))
    betahat1 = pd.DataFrame(np.zeros((length, k_dim)))
    sigma_data = [[n, np.zeros((k_dim, k_dim))] for n in range(length)]
    sigma2hat1 = pd.DataFrame(sigma_data).drop(0, axis=1)

    t0_1 = time.time()
    for t in np.arange(100, length - 1):
        if t == 100:

            rls_instance = RecursiveOLS(y.iloc[:t + 1, :].values,
                                        X.iloc[:t + 1, :].values)
            # add 0 to t inclusive
            beta_h = rls_instance.beta
        else:
            # _run_update only with last values
            rls_instance.update(y.iloc[t, :].values,
                                X.iloc[t, :].values)
            # add t onto 0 to t-1 inclusive
            beta_h = rls_instance.beta

        betahat1.iloc[t + 1, :] = beta_h.T
        # sigma2hat1.iloc[t+1,0] = p
        # create forecast and make scalar
        x_new = X.iloc[t + 1, :]  # need _slice2col (recast type) if using matrix mult
        # yhat1[t+1] = (beta.T @ x_new)[0,0]
        temp = 0
        for i in range(k_dim):  # beta is col vec
            temp += beta_h[i][0] * x_new[i]
        yhat1.iloc[t + 1, 0] = temp

    t1_1 = time.time()
    print('Python RLS: {}'.format(t1_1 - t0_1))  # 58 secs

    assert ((betahat1 - betahat) < 10e-10).all().all()
    return
