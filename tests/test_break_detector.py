import unittest

import numpy as np
import pandas as pd
# from strat_macro_qrv.models.recursive_ls.rls.ewrls import EWRLSRidge, EWRLSChangePoint


# from strat_macro_qrv.models.recursive_ls.tests.test_break_detector import (
#     jiang_zhang, simulate_and_store,
#     simulate_and_store_breaks
#     )
from ewrls.ewrls import EWRLSRidge


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()


def changing_means():
    # generate series with changing means
    # x1 at 5000, 12000, x2 at 15000,

    # var at 12000
    eps_x = np.convolve(np.random.randn(20000) * 0.01, np.ones(5))[:20000] / 5  # 5 day movav - adds 4 days data
    eps_x2 = np.convolve(np.random.randn(20000) * 0.01, np.ones(10))[:20000] / 10
    X1 = np.append(np.append(np.ones((5000, 1)) * 0.5, np.ones((7000, 1)) * 1.3), np.ones((8000, 1)) * 0.2) + eps_x
    X1 = X1[:, np.newaxis]
    X2 = np.append(np.ones((15000, 1)) * 1, np.ones((5000, 1)) * (-3)) + eps_x2
    X2 = X2[:, np.newaxis]
    X = np.append(X1, X2, axis=1)
    beta = np.random.uniform(low=0.5, high=1.2, size=2)[:, np.newaxis]
    eps = np.append(np.random.randn(12000) * 0.5, np.random.randn(8000) * 0.3)[:, np.newaxis]
    y = X @ beta + eps
    y = pd.Series(y.reshape(20000, ), index=range(len(y)))
    X = pd.DataFrame(X, index=y.index, columns=['x1', 'x2'])
    beta = pd.Dataframe(np.repeat(beta.T, 20000, axis=0), index=y.index, columns=['beta1', 'beta2'])
    return X, y, beta


def changing_means_betas():
    # generate series with changing means
    # x1 at 5000, 12000, x2 at 15000,
    # beta at 10000
    # var at 12000
    eps_x = np.convolve(np.random.randn(20000) * 0.01, np.ones(5))[:20000] / 5  # 5 day movav - adds 4 days data
    eps_x2 = np.convolve(np.random.randn(20000) * 0.01, np.ones(10))[:20000] / 10
    X1 = np.append(np.append(np.ones((5000, 1)) * 0.5, np.ones((7000, 1)) * 1.3), np.ones((8000, 1)) * 0.2) + eps_x
    X1 = X1[:, np.newaxis]
    X2 = np.append(np.ones((15000, 1)) * 1, np.ones((5000, 1)) * (-3)) + eps_x2
    X2 = X2[:, np.newaxis]
    X = np.append(X1, X2, axis=1)
    beta1 = np.random.uniform(low=0.5, high=1.2, size=2)[:, np.newaxis]
    beta2 = np.random.uniform(low=-0.2, high=0.5, size=2)[:, np.newaxis]
    beta = np.append(np.repeat(beta1.T, 10000, axis=0), np.repeat(beta2.T, 10000, axis=0), axis=0)

    eps = np.append(np.random.randn(12000) * 0.2, np.random.randn(8000) * 0.3)[:, np.newaxis]
    y1 = X[:10000] @ beta1 + eps[:10000]
    y2 = X[10000:] @ beta2 + eps[10000:]
    y = np.append(y1, y2)
    y = pd.Series(y.reshape(20000, ), index=range(len(y)))
    X = pd.DataFrame(X, index=y.index, columns=['x1', 'x2'])
    beta = pd.DataFrame(beta, index=y.index, columns=['beta1', 'beta2'])
    return X, y, beta


def jiang_zhang(sigma_v_sqr=0.01):
    '''
    From A Novel variable-length sliding window blockwise least-squares algorithm for
    on-line estimation of time-varying parameters, Intl J Adapt Control and Signal Proc 2004: 18:505-521
    Jin Jiang and Youmin Zhang
    @param sigma_v_sqr:
    @type sigma_v_sqr:
    @return:
    @rtype:
    '''
    # arma(2,2) with jumping parameters

    sigma_v = sigma_v_sqr ** (0.5)
    v = np.random.randn(1500) * sigma_v
    u = np.random.randn(1500)
    # original coefs in paper, probably a mistake in 3rd row
    # coefs = np.array([[-1.5, 0.7, 1., 0.5],
    #                   [1E-3, -6.7E-4, -1E-3, -6.7E-4],
    #                   [-1.2E-3, 8E-4, 1.2E-3, 8E-4],
    #                   [-1.5, 0.7, 1.0, 0.5],
    #                   [-1.2, 0.5, 0.7, 0.3]])

    coefs = np.array([[-1.5, 0.7, 1., 0.5],
                      [1E-3, -6.7E-4, -1E-3, -6.7E-4],
                      [-2E-3, 1.34E-3, 2E-3, 1.34E-3],
                      [-1.5, 0.7, 1.0, 0.5],
                      [-1.2, 0.5, 0.7, 0.3]])

    interval_lengths = [200, 300, 150, 350, 500]
    list_of_coefs = [np.repeat(coefs[0:1, :], 200, axis=0),
                     coefs[0:1, :] +
                     np.repeat(np.arange(300)[:, np.newaxis], 4, axis=1) \
                     * np.repeat(coefs[1:2, :], 300, axis=0),
                     coefs[0:1, :] + coefs[1:2, :] * interval_lengths[1] +
                     np.repeat(np.arange(150)[:, np.newaxis], 4, axis=1) \
                     * np.repeat(coefs[2:3, :], 150, axis=0),

                     np.repeat(coefs[3:4, :], 350, axis=0),
                     np.repeat(coefs[4:, :], 500, axis=0)]

    coefficients = np.concatenate(list_of_coefs)

    y = np.zeros((1500, 1))
    z = np.zeros((1500, 1))
    X = np.zeros((1500, 4))
    y[0] = 0
    y[1] = 0
    z[0] = v[0]
    z[1] = v[1]
    for t in range(2, 1500):
        row = coefficients[t, :]
        y[t] = -row[0] * y[t - 1] - row[1] * y[t - 2] + row[2] * u[t - 1] + row[3] * u[t - 2]
        X[t, :] = np.hstack([-z[t - 1], -z[t - 2], u[t - 1], u[t - 2]])
        z[t] = y[t] + z[t]

    z = pd.Series(z.reshape(len(z), ), index=range(len(z)), name='y')
    X = pd.DataFrame(X, index=z.index, columns=['-z(t-1)', '-z(t-2)', 'u(t-1)', 'u(t-2)'])
    beta = pd.DataFrame(coefficients, index=z.index, columns=['a1', 'a2', 'b1', 'b2'])
    return X, z, beta


def random_beta_jump(k_dim, length):
    '''
    Random betas, random features. One jump in beta.
    '''
    x = np.ones((5000, 1)) * 0.5
    scalings = np.random.uniform(low=0.8, high=1.4, size=k_dim)
    X = pd.DataFrame(np.random.randn(k_dim * length).reshape((length, k_dim))). \
        mul(scalings, axis=1)
    beta1 = np.random.uniform(low=0.8, high=3, size=k_dim)[:, np.newaxis]
    beta2 = np.random.uniform(low=0.8, high=3, size=k_dim)[:, np.newaxis]
    beta = np.append(np.repeat(beta1.T, 15000, axis=0), np.repeat(beta2.T, 15000, axis=0), axis=0)
    # np.array([[1.2, 0.7]]).T
    eps = pd.DataFrame(np.random.randn(length) * 0.5)
    y1 = X.loc[:15000, :].dot(beta1) + eps.loc[:15000]
    y2 = X.loc[15000:, :].dot(beta2) + eps.loc[15000:]
    y = np.concatenate(y1, y2, axis=0)
    y = pd.Series(y, index=range(len(y)), name='y')
    X.columns = ['var_' + str(k) for k in range(k_dim)]
    beta_col = ['beta_' + str(j) for j in range(k_dim)]
    beta = pd.DataFrame(beta, index=y.index, columns=beta_col)
    return X, y, beta


def simulate_and_store(simulation_func, params=None, span=40, regularisation=5):
    if params == None:
        X, y, beta_true = simulation_func()
    else:
        X, y, beta_true = simulation_func(params)

    y.name = 'y'
    # recursive_ls = EWRLSRidge(num_features=X.shape[1], span=100, regularization=0.0, forecast_history=True)
    # recursive_ls.update(y.values,X.values)
    rls2 = EWRLSRidge(num_features=X.shape[1], span=span, regularization=regularisation)
    rls2.update(y.values, X.values)
    # preds = pd.concat([y, rls2.concatenate_results()], axis=1)
    preds = rls2.concatenate_results()
    beta_cols = ['beta_' + str(k) for k in range(beta_true.shape[1])]
    pd.concat([pd.DataFrame(rls2.beta_history.T, columns=beta_cols),
               beta_true], axis=1).to_csv('betas.csv')
    preds.to_csv('Predictions.csv')
    rls2.printstate()


def simulate_and_store_breaks(simulation_func, params=None, span=40, regularisation=5, breaks=[[]]):
    if params == None:
        X, y, beta_true = simulation_func()
    else:
        X, y, beta_true = simulation_func(params)

    y.name = 'y'
    # recursive_ls = EWRLSRidge(num_features=X.shape[1], span=100, regularization=0.0, forecast_history=True)
    # recursive_ls.update(y.values,X.values)

    exp_model = EWRLSChangePoint(num_features=X.shape[1], span=span, regularization=regularisation,
                                 init_change_times=breaks, overlap=10)
    exp_model.update(y.values, X.values)

    preds = pd.concat([y, exp_model.concatenate_results()], axis=1)

    beta_cols = ['beta_' + str(k) for k in range(beta_true.shape[1])]
    pd.concat([pd.DataFrame(exp_model.beta_history_total.T, columns=beta_cols),
               beta_true], axis=1).to_csv('betas_break.csv')
    preds.to_csv('Predictions_break.csv')
    exp_model.printstate()


def main():
    length = 30000
    k_dim = 2

    simulate_and_store(jiang_zhang, 0.1, span=40, regularisation=2)
    simulate_and_store_breaks(jiang_zhang, span=40, regularisation=2, breaks=[500, 1000])

    # simulate_and_store(changing_means, None, span=50)
    # x = np.array([np.random.randn(10000)])
    # z = np.array([np.random.randn(10000) * 1.5])

    # y = 1.2 * x + 0.7 * z  + np.random.randn(10000)*0.05

    # bench_ols_rls1()
    # bench_ols_rls2()

    #
    # X, y = random_beta_jump(k_dim, length)
    #
    # recursive_ls = EWRLSRidge(num_features=X.shape[1], span=20, regularization=0.0, forecast_history=True)
    # recursive_ls.update(y.values,X.values)
    # bench_ols_rls1()

    # X, y, beta_true = changing_means_betas()
    # y.name = 'y'
    # # recursive_ls = EWRLSRidge(num_features=X.shape[1], span=100, regularization=0.0, forecast_history=True)
    # # recursive_ls.update(y.values,X.values)
    # rls2 = EWRLSRidge(num_features=X.shape[1], span=2000, regularization=5, forecast_history=True)
    # rls2.update(y.values, X.values)
    #
    # preds = pd.concat([y,
    #                    pd.Series(rls2.y_hat_t_tminus1, index=y.index, name='E[y(t)|t-1]'),
    #                    pd.Series(rls2.y_hat_t_t[:-1], name='E[y(t)|t]', index=y.index),
    #                    pd.Series(rls2.prediction_error_t_tminus1, index=y.index, name='prediction_error'),
    #                    pd.Series(rls2.y_hat_var, name='Var[y(t)|t-1]', index=y.index),
    #                    pd.Series(rls2.y_var, name = 'Var[y(t)]  Exp Wtd'),
    #                    pd.Series(rls2.residual_var, name='Var(forecast_error)]')], axis=1)
    #
    # pd.concat([pd.DataFrame(rls2.beta_history.T), pd.DataFrame(beta_true)], axis=1).to_csv('betas.csv')
    # preds.to_csv('Predictions.csv')
    # rls2.printstate()


if __name__ == '__main__':
    main()
