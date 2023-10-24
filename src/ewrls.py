import sys
import warnings

import numpy as np
import pandas as pd
from skopt import gp_minimize, gbrt_minimize, forest_minimize
import statsmodels.api as sm
from ewrls.ewma import ewma_vectorized_safe
from typing import Callable, Tuple
import constants as const

from utilities.settings import LOGGER

EPS = sys.float_info.epsilon
EPS2 = 1 - np.nextafter(1., 0.)
TOL = 1E-10
SPAN_TOL = 10
LAMBDA_TOL = 1E-7
VERBOSE = False


class EWRLSRidge(object):
    NORM_MAX = 3.0

    def __init__(self, num_features: int, span: float, regularization: float, history: bool = False,
                 scale_l2: bool = True, calc_stderr: bool = False, beta_start=np.array([])):
        # save for possible reset

        self.num_features = num_features
        self.span = span
        self.regularization = regularization
        self.history = history
        self.scale_l2 = scale_l2
        self.calc_stderr = calc_stderr
        if not scale_l2:
            raise NotImplementedError('Rescaling L2 Penalty is not implemented yet')
        self.beta_start = beta_start
        # SPAN_TOL and LAMBDA_TOL to prevent instability
        regularization = max(regularization, LAMBDA_TOL)
        span = max(span, 1.0 + SPAN_TOL)
        if span < np.inf:
            alpha = 2.0 / (1.0 + span)  # alpha = smoothing factor, for span>=1
            forgetting_factor = 1.0 - alpha  # exponential wt
        else:
            forgetting_factor = 1.0  # Ordinary RLS
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization

        self.beta = StateVariable(name="beta", save_history=self.history)
        if len(beta_start) == 0:
            self.beta.update(np.zeros((num_features, 1)))
        else:
            self.beta.update(beta_start)
        if len(self.beta.state) != num_features:
            LOGGER.warn('Beta_start vector must have same dim as num_features')
            raise FloatingPointError('Inconsistent number of features')

        self.stderr = StateVariable(name="stderr", save_history=self.history)
        self.tstat = StateVariable(name="tstat", save_history=self.history)
        if self.calc_stderr:
            self.stderr.update(np.zeros((num_features, 1)), save=True)
            self.tstat.update(np.zeros((num_features, 1)), save=True)

        self.p = (1.0 / self.regularization) * np.eye(num_features)
        # start with small pos def matrix? Note for first k obs Sigma will be rank deficient
        # otherwise and p = Sigma^(-1)
        self._initialize_predictions()

    def __repr__(self):
        return f'EWRLSRidge("span={self.span}", "l2-regularization={self.regularization}",features={self.num_features})'

    def derive_span(self):
        alpha = (1 - self.forgetting_factor)
        if alpha > 0:
            span = (2 - alpha) / alpha
        else:
            span = np.inf
        return span

    def reset(self, span=None, regularization=None):
        if span is None:
            span = self.span
        if regularization is None:
            regularization = self.regularization
        # TODO: EWRLSRidge here
        self.__init__(num_features=self.num_features, span=span, regularization=regularization, history=self.history,
                      scale_l2=self.scale_l2, beta_start=self.beta_start)

    def cross_validation(self, y_train, x_train,
                         verbose=False,
                         span_bounds=const.span_bounds,  # (1 + SPAN_TOL, 5E10),
                         ridge_reg_bounds=const.ridge_reg_bounds,  # (LAMBDA_TOL, 1E3),
                         window=np.inf):
        # save the data
        self.y_train = y_train
        self.x_train = x_train
        self.cross_validation_verbose = verbose
        self.original_span = self.span
        self.original_regularization = self.regularization

        def cv_objective(self, span=None, regularization=None):
            # default to original span and regularization
            nobs = len(self.y_train)
            self.reset(span=span, regularization=regularization)

            if (window < np.inf):
                cutoff = max(nobs - window, 0)
                self.update(self.y_train[:cutoff], self.x_train[:cutoff, :])
                sse1 = self.sse
                nobs1 = self.nobs
                self.update(self.y_train[cutoff:], self.x_train[cutoff:, :])
                sse2 = self.sse
                nobs2 = self.nobs
                windowed_mse = (sse2 - sse1) / (nobs2 - nobs1)
                self.windowed_mse = windowed_mse
            else:
                self.update(self.y_train, self.x_train)
                windowed_mse = self.mse
                self.windowed_mse = windowed_mse
            # TODO: Find out when and why returns array!?!
            if isinstance(windowed_mse, np.ndarray):
                windowed_mse = windowed_mse[0]
            if self.cross_validation_verbose:
                LOGGER.info('Span = {}'.format(span) + ' Reg = {}'.format(regularization) +
                            ' MSE = {}'.format(windowed_mse))
            return windowed_mse

        warnings.filterwarnings('ignore', '.*objective has been.*', )
        res = gp_minimize(lambda x: cv_objective(self, x[0], x[1]),  # the function to minimize
                          [span_bounds, ridge_reg_bounds],  # the bounds on each dimension of x
                          x0=[self.original_span, self.original_regularization],  # the starting point
                          acq_func="gp_hedge",  # the acquisition function (optional)
                          n_calls=150,  # the number of evaluations of f including at x0
                          n_random_starts=15,  # the number of random initial points
                          random_state=789)
        self.cv_results = res
        # don't leave at last evaluated point, reset to optimal
        self.reset(span=res.x[0], regularization=res.x[1])
        if verbose:
            LOGGER.info('Optimal CV Error at span = {}'.format(self.span) +
                        ' reg = {}'.format(self.regularization) +
                        ' mse = {}'.format(self.windowed_mse)
                        )

    def _initialize_predictions(self):
        self.sse = 0.0
        self.mse = np.nan
        self.mse_adj = np.nan
        self.windowed_mse = np.nan
        self.nobs = 0
        self.nobs_total = 0
        self.y = StateVariable(name="y", col_header="y(t)",
                               method='normal',
                               save_history=True)
        self.prediction_error_t_tminus1 = StateVariable(name="prediction_error_t_tminus1",
                                                        col_header="y(t)-E[y(t)|t-1]",
                                                        forgetting_factor=self.forgetting_factor,
                                                        save_history=True,
                                                        method='normal')
        self.y_hat_t_tminus1 = StateVariable(name="y_hat_t_tminus1",
                                             col_header="E[y(t)|t-1]",
                                             forgetting_factor=self.forgetting_factor,
                                             save_history=True,
                                             method='normal')
        # y_hat_t_tminus1 = E[y(t)| beta(t-1),X(t)]  Start empty
        self.y_hat_t_t = StateVariable(name="y_hat_t_t",
                                       col_header="E[y(t)|t]",
                                       forgetting_factor=self.forgetting_factor,
                                       save_history=True,
                                       method='normal')
        # y_hat_t_t = E[y(t)| beta(t), X(t)] NOT start with nan (?)

        self._initialize_other_state()

    def _initialize_other_state(self):
        self.y_mean = StateVariable(name='y_mean',
                                    method='ewm',
                                    forgetting_factor=self.forgetting_factor,
                                    save_history=self.history)
        # endogenous (exp wtd) mean
        self.y_var = StateVariable(name='y_var',
                                   col_header="Var[y(t)] Exp Wtd",
                                   method='ewm',
                                   forgetting_factor=self.forgetting_factor,
                                   save_history=self.history)
        # endogenous (exp wtd) variance
        self.y_hat_mean = StateVariable(name='y_hat_mean',
                                        method='ewm',
                                        forgetting_factor=self.forgetting_factor,
                                        save_history=self.history)
        # needed to calc variance
        self.y_hat_var = StateVariable(name="y_hat_var",
                                       col_header="Var[y(t)|t-1]",
                                       method='ewm',
                                       forgetting_factor=self.forgetting_factor,
                                       save_history=self.history)
        # forecast (y_hat_t_tminus1) variance
        self.residual_var = StateVariable(name='redisual_var',
                                          col_header="Var(forecast_error)",
                                          method='ewm',
                                          forgetting_factor=self.forgetting_factor,
                                          save_history=self.history)
        # variance OOS prediction (t_tminus1)
        self.in_sample_std = StateVariable(name='in_sample_std',
                                           method='std',
                                           forgetting_factor=self.forgetting_factor,
                                           save_history=self.history)

    def set_start(self, p, beta=np.array([]), nobs=0):
        self.p = p
        self.beta.update(beta)
        self.nobs = nobs
        self.nobs_total = nobs
        if len(self.beta) == 0:
            self.beta.update(np.zeros((self.k_dim, 1)))

    def _run_update(self, y_t: np.ndarray,
                    x_t: np.ndarray, save=True):
        '''
        y_t is 1-series of dependent data (1 obs)
        x_t is k-series of independent features (1 obs)
        beta is previous obs beta np.array (1,k) - col vec
        p is previous obs X'X np.array (k,k)
        don't transform x_t and y_t but make sure
        beta and p are correct dimensions at the end
        y_t is (1,) and x_t is (k,)
        _run_update last period's ex-ante-forecast by default (if
        _run_update is used as a one-off). Note that within bulk-_run_update, the ex-ante
        forecasts are updated outside of this function.
        '''

        '''
        Gain quantities
        '''
        x_t = self._column_vector(x_t)
        y_t = self._array2float(y_t)
        if len(x_t) != len(self.beta.state):
            LOGGER.warn('_run_update data is of wrong dimension')
            raise FloatingPointError

        # saved_handler = np.seterrcall(self.err_handler)
        # save_err = np.seterr(over='call', under='ignore', divide='call')

        k_gain, norm_factor = self._gain(x_t)
        # generate forecast before updating beta.
        # y_hat_t_tminus1 is E[y(t) | x(t), y(1:t-1)]
        #    = beta[x[1:t-1],y[1:t-1]] * x[t]
        y_hat_t_tminus1 = self.generate_prediction(x_t)
        # y_hat_t_tminus1 = x_t beta_t_tminus1
        # note beta_t_tminus1 = beta_(t-1)_(t-1) (no transition matrix)
        prediction_error_t_tminus1 = (y_t - y_hat_t_tminus1)
        scale_p = (1 / self.forgetting_factor) if self.scale_l2 else 1
        self.p = (scale_p * self.p - k_gain * k_gain.T * norm_factor)
        #    p - p.dot(x_t) * c * x_t.T.dot(P)
        self.beta.update(self.beta.state + k_gain * prediction_error_t_tminus1, save=save)
        # kx1 vector beta_t_t
        y_hat_t_t = self.generate_prediction(x_t, ndarray=False)
        self._update_data_vec(y_t, x_t, save=save)
        self._update_remaining_state_vars(y_hat_t_tminus1=y_hat_t_tminus1,
                                          y_hat_t_t=y_hat_t_t,
                                          prediction_error_t_tminus1=prediction_error_t_tminus1,
                                          save=save)

        return

    def _update_stderr(self):
        # uise mse or mse_adj ?
        self.stderr.update(np.sqrt(self.mse) * np.sqrt(np.diag(self.p)), save=True)
        # var(beta) = sigma^2 (X'X)^(-1)
        self.tstat.update((self.beta.state * self.stderr.state).T, save=True)

    def _update_data_vec(self, y_t, x_t, save=True):
        # ignore x_t in this method
        self.nobs += 1
        self.nobs_total += 1
        self.y.update(y_t, save=save)
        self.y_mean.update(y_t, save=save)
        self.y_var.update((y_t - self.y_mean.state) ** 2, save=save)

    def generate_prediction(self, x_vec, ndarray=False):
        '''
        Return current prediction using feature vector x_vec
        @param x_vec: updated feature
        @type x_vec: np.ndarray (column vector, row vector or vector)
        @param ndarray: return an array if true, return float if false
        @type ndarray: bool
        @return: prediction
        @rtype: float or ndarray
        '''
        if len(np.shape(x_vec, )) == 1:
            x_vec = self._column_vector(x_vec)
            pred = (x_vec.T @ self.beta.state)[0]
            # may be a float or an array
            if (~ndarray) and (type(pred) is np.ndarray):
                pred = pred[0]
        else:  # compare to my frozen extensions!!!
            if x_vec.shape[0] > x_vec.shape[1]:
                # print('Transposed x_vec in EWRLSRidge prediction')
                x_vec = x_vec.T
            else:
                pass
                # print('did not transpose x-vec')
            # todo: why is x_vec sometimes row vec and sometimes col vec
            pred = (x_vec @ self.beta.state)[0, 0]
        return pred

    def _gain(self, x_update: np.ndarray) -> np.ndarray:
        '''
        Kalman Gain, using priors
        @param x_update: new data vector
        @type x_update: np.array() (num_features,1)
        @return: k_gain - kalman _gain
        @rtype: np.float
        '''
        norm_factor = (1 + (1 / self.forgetting_factor) *
                       x_update.T @ self.p @ x_update)
        k_gain = (1 / self.forgetting_factor) * (self.p @ x_update) / norm_factor
        # (num_features,1)  vector
        # x_t is a (num_features,) vector, so cannot .T it
        return k_gain, norm_factor

    def _update_remaining_state_vars(self,
                                     y_hat_t_tminus1,
                                     y_hat_t_t,
                                     prediction_error_t_tminus1, save=True):
        '''

        @param y_hat_t_tminus1:  E[y(t) | beta(t-1), x(t)].
        Ex-ante is E[y(t+1) | beta(t), x(t+1)]. Generated using beta(t-1)
        @type y_hat_t_tminus1:
        @param prediction_error_t_tminus1:
        @type prediction_error_t_tminus1:
        @param x_t:  updated exogenous variable
        @type x_t: pd.array() column vector
        @param y_t: updated endogenous data
        @type y_t: pd.array() or float?
        @return: None
        @rtype:
        '''
        self.sse += prediction_error_t_tminus1 ** 2
        self.mse = self.sse / self.nobs
        self.mse_adj = (self.sse / (self.nobs - self.num_features) if self.nobs > self.num_features else np.nan)
        # regenerate forecast, after having updated beta
        # E[y(t) | x(t), beta(t)] where beta(t) has already been updated (and seen y(t))
        # y_hat_t_tminus1 is E[y(t) | x(t), y(1:t)]
        #    = beta[x[1:t],y[1:t]] * x[t] (fully in-sample)

        self.prediction_error_t_tminus1.update(prediction_error_t_tminus1, save=save)
        self.y_hat_t_tminus1.update(y_hat_t_tminus1, save=save)
        self.y_hat_t_t.update(y_hat_t_t, save=save)
        self.y_hat_mean.update(y_hat_t_tminus1, save=save)
        self.y_hat_var.update((y_hat_t_tminus1 - self.y_hat_mean.state) ** 2, save=save)
        # residual has zero mean
        self.residual_var.update(prediction_error_t_tminus1 ** 2, save=save)
        self.in_sample_std.update(prediction_error_t_tminus1, save=save)
        if self.calc_stderr:
            self._update_stderr()

    # @staticmethod
    def normalize_data(self, exog, min_sample_length: int = 300, expanding: bool = False,
                       norm_max: float = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
         create zspreads using min-sample-length mean and std
         if expanding, then in-sample periuod (with loookahead) for min_sample_length
        @param exog:  data to be noramlized
        @type exog:  pd.DataFrame,pd.DataSeries
        @param min_sample_length:  in-sample period (use to create norm factors, or if expanding=-True, then make constant
           mean and stde over this period
        @type min_sample_length:
        @param expanding:  if expanding then use expanding window
        @type expanding:  Boolean
        @return: normalized data, mean and stdev
        @rtype:  dataframe or series, for exog data, float and float for mean and stdev
        '''
        if norm_max == 0:
            norm_max = self.NORM_MAX
        dataframe = False  # should be a Class parameter
        if type(exog) == pd.DataFrame:
            dataframe = True

        if (type(exog) == pd.DataFrame) and ('const' in exog.columns):
            exog_non_const = exog.drop(columns=['const'])
            const = True
        else:  # series can't remove constant
            exog_non_const = exog
            const = False
        min_sample_length = max(min_sample_length, 1)

        if expanding:
            exog_stdev = exog_non_const.expanding(min_periods=min_sample_length).std()
            exog_mean = exog_non_const.expanding(min_periods=min_sample_length).mean()
            if dataframe:
                exog_mean.iloc[:min_sample_length, :] = np.nan
                exog_stdev.iloc[:min_sample_length, :] = np.nan
            else:
                exog_mean.iloc[:min_sample_length] = np.nan
                exog_stdev.iloc[:min_sample_length] = np.nan
            exog_mean = exog_mean.fillna(method='bfill')
            exog_stdev = exog_stdev.fillna(method='bfill')
            # no NANs!
            # set to 'in-sample/' during first period
            # should nan out first [1:min_periods] on BOTH mean and stdev and then bfill
        else:  # normal, test period and others
            if min_sample_length < np.inf:
                if dataframe:
                    exog_stdev = exog_non_const.iloc[:min_sample_length, :].std()
                    exog_mean = exog_non_const.iloc[:min_sample_length, :].mean()
                else:  # series
                    exog_stdev = exog_non_const.iloc[:min_sample_length].std()
                    exog_mean = exog_non_const.iloc[:min_sample_length].mean()
            else:  # == np.inf
                exog_stdev = exog_non_const.std()
                exog_mean = exog_non_const.mean()
        exog_norm = ((exog_non_const - exog_mean).div(exog_stdev)).clip(-1 * norm_max, norm_max)
        # exog_norm.loc[exog_norm.map(lambda r: ~np.isfinite(r))] = 0.0
        exog_norm = exog_norm.fillna(0.0)
        # exog_norm[~np.isfinite(exog_norm)] = 0

        if const:
            exog_norm = pd.concat([exog['const'], exog_norm], axis=1)  # put the constant back in
        # how can this still be triggered?
        # if exog_norm.applymap(np.isinf).any().any():
        #     exog_norm[~np.isfinite(exog_norm)] = 0
        #     LOGGER.warn('We still had an inf after capping normalization')
        # remove possible inf/-inf AND np.nan
        return exog_norm, exog_stdev, exog_mean

    @staticmethod
    def renormalize_data(normalized_features, exog_stdev, exog_mean):

        const = False
        if (type(normalized_features) is pd.DataFrame) and ('const' in normalized_features.columns):
            normalized_features = normalized_features.drop(columns='const')
            const = True
        renorm = normalized_features * exog_stdev + exog_mean
        if const:
            renorm = pd.concat([normalized_features['const'], renorm], axis=1)
        if renorm.map(np.isinf).any():
            renorm[~np.isfinite(renorm)] = 0
            LOGGER.warn('We still had an inf during data renormalization')

        return renorm

    @staticmethod
    def renormalize_betas(betas, exog_stdev, exog_mean):  # not usedg

        betas_const = betas['const']
        betas_nonconst = betas.drop('const')
        betas_nonconst_rescaled = betas_nonconst.div(exog_stdev)
        mean_adjust = betas_nonconst.mul(exog_mean).div(exog_stdev).sum()
        betas_const -= mean_adjust
        betas_rescaled = pd.Series(betas_const, index=['const'], name='beta')
        betas_rescaled = betas_rescaled.combine_first(betas_nonconst_rescaled)
        # concatenate axis=0 for series
        return betas_rescaled

    @staticmethod
    def _vec_convert(y):
        if (type(y) == np.float64) or (type(y) == float):
            # convert to (1,) array
            y = np.array([y])
        if (type(y) == np.ndarray and np.ndim(y) > 1
                and np.min(np.shape(y)) == 1):
            if y.shape[0] == 1:
                y = y[0, :]
            elif y.shape[1] == 1:
                y = y[:, 0]

        return y

    @staticmethod
    def _array_convert(x):
        if len(np.shape(x)) == 1:
            # convert to (1,:) array
            x = np.array([x])
        return x

    def update(self, y: np.ndarray, x: np.ndarray) -> None:
        ''' Update single obs or do bulk update '''
        x = self._array_convert(x)
        y = self._vec_convert(y)
        new_nobs = len(y)
        # self.y_hat_t_t. append list(np.nan * np.ones(new_nobs)))  #
        for t in range(new_nobs):  # t<new_nobs-1
            y_slice = y[t]
            x_slice = self._column_vector(x[t, :])
            self._run_update(y_slice, x_slice, save=True)
        return

    def printstate(self, log=True):
        formatted_str = 'Current states\n'
        span = self.derive_span()
        formatted_str += "self.span = {}\n".format(span)
        formatted_str += "self.regularization ={}\n".format(self.regularization)
        formatted_str += "self.beta  = {}\n".format(self.beta.state)
        formatted_str += "self.p  = {}\n".format(self.p)
        formatted_str += "self.nobs = {}\n".format(self.nobs)
        formatted_str += "self.sse = {}\n".format(self.sse)
        if log:
            LOGGER.info(formatted_str)
        else:
            print('Logger not on - EWRLS')
            print(formatted_str)

    def concatenate_results(self):
        stored_index = np.arange(len(self.y_hat_t_tminus1.total_history))

        if self.history:
            concat_list = [self.y,
                           self.y_hat_t_tminus1,
                           self.y_hat_t_t,
                           self.prediction_error_t_tminus1,
                           self.y_hat_var,
                           self.y_var,
                           self.residual_var]
        else:
            concat_list = [self.y,
                           self.y_hat_t_tminus1,
                           self.prediction_error_t_tminus1]

        return pd.concat([x.create_ser(index=stored_index) for x in concat_list], axis=1)

    def err_handler(self, type, flag):
        LOGGER.warn('Runtime Numerical Warning\n')
        LOGGER.warn("Floating point error {} ".format(type) + " with flag {}".format(flag))
        self.printstate(log=True)
        LOGGER.warn('It is likely that this is an overflow error. \n'
                    'Previously we could increase LAMBDATOL to prevent.\n'
                    'Continued issues require either OLS or stabilization method')
        raise np.TooHardError('Stop now!')

    @staticmethod
    def _slice2col(x: pd.Series) -> np.ndarray:
        return x.values[:, np.newaxis]

    @staticmethod
    def _column_vector(x_update):
        if len(x_update.shape) > 1:
            if (x_update.shape[0] > 1) and (x_update.shape[1] > 1):
                raise FloatingPointError('Supplied full matrix, where col vector, '
                                         'row vector or vector needed')
            elif x_update.shape[0] == 1:
                x_update = x_update.T  # transform into (k,1) vector
            elif x_update.shape[1] == 1:
                pass  # already col vector
        if len(x_update.shape) == 1:
            x_update = x_update[:, np.newaxis]
            # change (k,) into (k,1) i.e., column vector
        return x_update

    def _array2float(self, y) -> np.float:
        if type(y) is pd.Series:
            return y[0]
        if type(y) is pd.DataFrame:
            return y.iloc[0, 0]
        if type(y) is np.ndarray:
            return self._column_vector(y)[0, 0]
        else:  # float or int
            return y

    def forecast_vol(self, span=np.nan):
        if span == np.nan:
            span = self.derive_span()
        ex_ante_pred_series = self.y_hat_t_tminus1.create_ser().dropna()
        if span < np.inf:
            ex_ante_vol = ex_ante_pred_series.ewm(span=span).std().iloc[-1]
        else:
            ex_ante_vol = ex_ante_pred_series.std()
        self.ex_ante_vol = ex_ante_vol


class EWRLSChangePoint(EWRLSRidge):
    def __init__(self, num_features: float, span: float = 1000, regularization: float = 0.0,
                 init_change_times: list = [], change_detector_list: list = [], rw_detector_list: list = [],
                 overlap: int = 10, hot_start=True, history=False):
        super().__init__(num_features=num_features, span=span, regularization=regularization, history=history)
        self.RESTART_INT = 2
        self.change_date_state = False
        self.overlap = overlap
        if init_change_times != []:
            self.change_times = [x for x in init_change_times if x > self.overlap]
        else:
            self.change_times = []
        # Exactly 4 break detectors stat vs trailiing history
        self.detectordict = {
            'forecast_fit': self.forecast_fit, 'first_diff': self.first_diff,
            'second_diff': self.second_diff, 'beta_diff': self.beta_diff}
        if change_detector_list != []:
            self.change_detectors = [BreakDetector(name=func_str,
                                                   func=self.detectordict[func_str],
                                                   params=params)
                                     for func_str, params in change_detector_list]
        else:
            self.change_detectors = []

        if rw_detector_list != []:
            self.rw_detectors = [BreakDetector(name=func_str,
                                               func=self.detectordict[func_str],
                                               params=params)
                                 for func_str, params in rw_detector_list]
        else:
            self.rw_detectors = []

        # no changes at start of sample
        self.hot_start = hot_start
        self.nobs_total = 0
        self.tests = dict()  # variable names - dict of dict
        self._initialize_changept_state()

    def reset(self):
        raise NotImplementedError

    def cross_validation(self):
        raise NotImplementedError('CV not implemented in EWRLSBatch yet')

    def _initialize_changept_state(self):
        # unique to EWRLSChange
        self.beta.set_save_history(True)
        self.cusumsqr = StateVariable(name='cusumsqr',
                                      forgetting_factor=1.,
                                      save_history=True,
                                      method='normal')
        self.x = StateVariable(name='x',
                               method='normal',
                               save_history=True)

    def _run_update(self, y_t: np.ndarray,
                    x_t: np.ndarray, save=True):
        super()._run_update(y_t, x_t, save=save)
        self.nobs_total += 1
        # TODO: Make recursive method here, save prediction_error_t_tminus1[-self.overlap]
        #  if nobs>windowsize and then delete entry
        length = max(self.nobs, self.overlap)
        self.cusumsqr.update(np.mean(self.prediction_error_t_tminus1.total_history[-length:] ** 2))

    def _update_data_vec(self, y_t, x_t, save=True):
        # may do ok with just past 10 days of x_t
        self.nobs += 1
        self.x.update(x_t, save=save)
        self.y.update(y_t, save=save)
        self.y_mean.update(y_t, save=save)
        self.y_var.update((y_t - self.y_mean.state) ** 2, save=save)

    def rw_model(self):
        # Use to bypass model output
        pass

    def update(self, y: np.ndarray, x: np.ndarray):
        x = super()._array_convert(x)
        y = super()._vec_convert(y)

        new_nobs = len(y)
        # self.y_hat_t_t. append list(np.nan * np.ones(new_nobs)))  #
        for t in range(new_nobs):  # t<new_nobs-1
            y_slice = y[t]
            x_slice = super()._column_vector(x[t, :])
            self.run_pre_update_tests()
            self._run_update(y_slice, x_slice, save=True)
            # self._save_history()  # copy over states before check for a break
            self.run_post_update_tests()
            self.decide_reset_model()

    def decide_reset_model(self):
        self.change_date_state = False
        index = self.nobs_total
        if (self.nobs_total in self.change_times) and (self.nobs > self.RESTART_INT * self.overlap):
            if self.overlap > index:
                # print('overlap = {}'.format(self.overlap), ' and obs ={}'.format(self.nobs_total))
                raise AttributeError('Cannot reset_total_history within overlap period')
            # make sure to include y[t] and have window_length correct
            y_overlap = self.y.total_history[index - self.overlap:index + 1]
            x_overlap = self.x.total_history[:, index - self.overlap:index + 1].T
            self.change_date_state = True
            self._reset_model(y_overlap, x_overlap)
            # do not overwrite cusumsqr_total for the overlapped data

    def report_change_times(self):
        return self.change_times

    def is_change_date(self):
        return self.change_date_state

    def run_post_update_tests(self):
        test_values = {}
        if self.change_detectors != []:
            change_values = {f.name: f() for f in self.change_detectors}
            test_values.update(change_values)
            truth_values = [f_val > 0 for f_name, f_val in change_values.items()]

            if any(truth_values) and (self.nobs > self.RESTART_INT * self.overlap):
                self.add_cut_times([self.nobs_total])
        self.tests.update({(self.nobs_total - 1): test_values})
        # TODO: self.nobs_total doesn't align, starts at 1.

    def run_pre_update_tests(self):
        test_values = {}
        if self.rw_detectors != []:
            # TODO: RW_Model (rename it?) at start of all zspread turns--missing features = detectors.
            rw_values = {f.name: f() for f in self.rw_detectors}
            test_values.update(rw_values)
            truth_values = [f_val > 0 for f_name, f_val in rw_values.items()]
            if any(truth_values) and (self.nobs > self.RESTART_INT * self.overlap):
                # TODO: Decide if 'outlier' do we need rest-time?
                self.rw_model()
                # rw model - ignore data and let yhat(t+1) = y(t)
                self.add_cut_times([self.nobs_total])

    def forecast_fit(self, param_tuple=(0.99, 3.5)):
        forget_factor, multiple = param_tuple
        sumsqr_diff_abs = np.abs((self.prediction_error_t_tminus1.total_history))
        ewstd = ewma_vectorized_safe(sumsqr_diff_abs ** 2, alpha=1 - forget_factor) ** 0.5
        return (sumsqr_diff_abs[-1] - multiple * ewstd[-1])

    def beta_diff(self, param_tuple=(0.99, 3.5)):
        forget_factor, multiple = param_tuple
        if self.beta.total_history.shape[1] == 1:
            return -1 * np.inf
        l2_norm_diff = np.linalg.norm(np.diff(self.beta.total_history), axis=0)  # default l2 norm
        ewstd = ewma_vectorized_safe(l2_norm_diff ** 2, alpha=1 - forget_factor) ** 0.5
        return (l2_norm_diff[-1] - multiple * ewstd[-1])

    def first_diff(self, param_tuple=(0.99, 3.5)):
        forget_factor, multiple = param_tuple
        sumsqr_diff_abs = np.abs(np.append([0.0] * 1, np.diff(self.y.total_history, 1)))
        ewstd = ewma_vectorized_safe(sumsqr_diff_abs ** 2, alpha=1 - forget_factor) ** 0.5
        return (sumsqr_diff_abs[-1] - multiple * ewstd[-1])

    def second_diff(self, param_tuple=(0.99, 3.5)):
        # TODO: Make sure ewma works with nan data!
        forget_factor, multiple = param_tuple
        sumsqr_diff_abs = np.abs(np.append([0.0] * 2, np.diff(self.y.total_history, 2)))
        ewstd = ewma_vectorized_safe(sumsqr_diff_abs ** 2, alpha=1 - forget_factor) ** 0.5
        return (sumsqr_diff_abs[-1] - multiple * ewstd[-1])

    def add_cut_times(self, new_change_times):
        current_cuts = self.change_times
        total_cuts = list(set(current_cuts).union(set(new_change_times)))
        total_cuts.sort()
        self.change_times = total_cuts

    def _reset_model(self, y_start, x_start):
        num_features = self.beta.state.shape[0]
        if self.hot_start:
            beta = self.beta.state  # most recent beta. Do we want to start there?
        else:
            beta = np.zeros((num_features, 1))
        # self.beta.set_save_history(True)
        '''
        Reset State variables and Histories
        '''

        update_list = [self.beta, self.y, self.prediction_error_t_tminus1,
                       self.y_hat_t_tminus1, self.y_hat_t_t,
                       self.y_mean, self.y_var, self.y_hat_mean, self.y_hat_var,
                       self.residual_var, self.in_sample_std, self.cusumsqr, self.x]

        for x in update_list:
            if x is self.beta:
                x.reset(state_0=beta)
            else:
                x.reset()
        self.nobs = 0
        self.restart_rls_overlap(x_start, y_start)

    def restart_rls_overlap(self, x_start, y_start):
        reset_nobs = y_start.shape[0]
        for t in range(reset_nobs):  # t<new_nobs-1
            y_slice = y_start[t]
            x_slice = super()._column_vector(x_start[t, :])
            super()._run_update(y_slice, x_slice, save=False)
            # don't _run_update nobs_total or .total_histories
        self.cusumsqr.update(np.mean(
            self.prediction_error_t_tminus1.total_history[-1 *
                                                          max(self.nobs,
                                                              self.overlap):] ** 2),
            save=False)
        # create final cusumsqr but only store history once it continues _run_update

    def _save_history(self):
        # and SAVE_Update self.nobs_total = previous _run_update, DO not append, but only append
        # from previous _run_update (need two vaiables. if nobs = current run, nobs_total now
        # is previous _run_update, then nobs_current must be set to current total
        pass

    def concatenate_results(self):
        # override parent class. Any way to just use parent code?
        preds = super().concatenate_results()
        # TODO: make index for self.y_hat etc so line up with breaks. For now index starts at 1!
        preds = pd.concat([preds, pd.DataFrame(self.tests).T], axis=1)
        # only missing endog (y)
        return preds


class BreakDetector:
    def __init__(self, name: str, func: Callable[[float, float], bool], params: Tuple[float, float]):
        self.name = name
        self.__func = func
        self.__params = params

    def __call__(self) -> float:
        return self.__func(self.__params)


class StateVariable(object):
    def __init__(self, name: str = None,
                 col_header: str = None,
                 forgetting_factor: float = 1.,
                 save_history: bool = False,
                 method: str = 'normal'):
        self.name = name
        self.forgetting_factor = forgetting_factor  # only used for ewm and ew_std
        if col_header is None:
            self.column_header = name
        else:
            self.column_header = col_header
        self.method = method  # ewm, et_std, normal
        self.save_history = save_history
        self.state = np.array([])
        self.history = np.array([])
        self.total_history = np.array([])

    def set_save_history(self, save_history: bool):
        self.save_history = save_history

    def update(self, value, save=True):
        # TODO: implement user-supplied function-based update
        if isinstance(value, int):
            value = float(value)
        if self.method == 'normal':
            self._norm_update(value)
        elif self.method == 'ewm':
            self._ewm_update(value)
        else:
            self._ew_std_update(value)
        if self.save_history:
            self.update_history(save=save)

    def reset_total_history(self, total_history):
        if self.save_history:
            self.total_history = total_history

    def reset(self, state_0=None):
        # reset state and history, leaving all else unch
        if state_0 is None:
            self.state = np.array([])
        else:
            self.state = state_0
        self.history = np.array([])

    def update_history(self, save=True):
        self._append_history(self.state)
        if save:
            self._append_total_history(self.state)
        # at beginning of break period, do not copy over state -> total_history

    def create_ser(self, index=None, name=None):
        # TODO: Create DataFrame for beta, etc
        if index is None:
            index = range(len(self.total_history))
        if name is None:
            name = self.column_header
        return pd.Series(self.total_history, name=name, index=index)

    def _save_variable(self, save_spot, var_name):
        '''
        https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
        can we take passed varname(?) and add _total or we only see local?
        @param varname:
        @type varname:
        @return:
        @rtype:
        '''
        # only copy over last value
        # var_name has a self.overlap overlap. Keep pre-break history
        # treat vectors and matrices differently.
        if save_spot.shape[0] == 0:
            save_spot = var_name.copy()
        elif len(var_name.shape) == 1:
            save_spot = np.append(save_spot, var_name[-1:])
        else:  # matrix
            save_spot = np.append(save_spot, var_name[:, -1:], axis=1)
        return save_spot

    def _norm_update(self, update_data):
        if type(update_data) == np.ndarray:
            update_data = self._column_vector(update_data)
        self.state = update_data

    def _ewm_update(self, update_data):
        if self.state is np.nan:
            self.state = update_data
        else:
            self.state = self.forgetting_factor * self.state + (1 - self.forgetting_factor) * update_data

    def _ew_std_update(self, update_data):
        if self.state is np.nan:
            self.state = np.abs(update_data)
        else:
            self.state = (self.forgetting_factor * self.state ** 2
                          + (1 - self.forgetting_factor) * update_data ** 2) ** (0.5)

    @staticmethod
    def _slice2col(x: pd.Series) -> np.ndarray:
        return x.values[:, np.newaxis]

    @staticmethod
    def _column_vector(x_update):
        if len(x_update.shape) > 1:
            if (x_update.shape[0] > 1) and (x_update.shape[1] > 1):
                raise FloatingPointError('Supplied full matrix, where col vector, '
                                         'row vector or vector needed')
            elif x_update.shape[0] == 1:
                x_update = x_update.T  # transform into (k,1) vector
            elif x_update.shape[1] == 1:
                pass  # already col vector
        if len(x_update.shape) == 1:
            x_update = x_update[:, np.newaxis]
            # change (k,) into (k,1) i.e., column vector
        return x_update

    @staticmethod
    def _vec_convert(y):
        if (type(y) == np.float64) or (type(y) == float):
            # convert to (1,) array
            y = np.array([y])
        return y

    # @staticmethod
    def _array_convert(self, x):
        if type(x) == np.ndarray:
            if np.ndim(x) == 1:
                # convert to (1,:) array
                x = np.array([x])
            self._column_vector(x)
        return x

    @staticmethod
    def _scalar_vec_mat(x):
        if type(x) in [np.float64, float]:
            return 'scalar'
        elif x.ndim == 1:
            if x.shape[0] == 1:
                return 'scalar'
            else:
                return 'vector'
        else:
            if np.prod(x.shape) == 1:
                return 'scalar'
            elif min(x.shape) == 1:
                return 'vector'
            else:
                return 'matrix'

    def _append_history(self, update_data):
        if self._scalar_vec_mat(update_data) == 'scalar':
            update_data = self._vec_convert(update_data)
        if self._scalar_vec_mat(update_data) == 'vector':
            update_data = self._column_vector(update_data)
        if len(self.history) == 0:
            self.history = update_data
        elif self._scalar_vec_mat(update_data) == 'scalar':
            self.history = np.append(self.history, update_data)
        else:
            self.history = np.append(self.history, update_data, axis=1)

    def _append_total_history(self, update_data):
        # TODO: Can we make _append_total_ with _append_history as same func to avoid repeat?
        if self._scalar_vec_mat(update_data) == 'scalar':
            update_data = self._vec_convert(update_data)
        if self._scalar_vec_mat(update_data) == 'vector':
            update_data = self._column_vector(update_data)
        if len(self.total_history) == 0:
            self.total_history = update_data
        elif self._scalar_vec_mat(update_data) == 'scalar':
            self.total_history = np.append(self.total_history, update_data)
        else:
            self.total_history = np.append(self.total_history, update_data, axis=1)


class EWRLSBatch(EWRLSRidge):

    def __init__(self, num_features: int, span: float = np.inf, regularization: float = 0.0,
                 history=False, scale_l2: bool = True, beta_start=np.array([])):
        super().__init__(num_features=num_features, span=span, regularization=regularization, history=history,
                         scale_l2=scale_l2, beta_start=beta_start)

        self.weights = pd.DataFrame()
        self.x = StateVariable(name='x', method='normal', save_history=True)

    def reset(self):
        raise NotImplementedError

    def cross_validation(self):
        raise NotImplementedError('CV not implemented in EWRLSBatch yet')

    def exp_weights(self, vec_length=None, span=None):
        if span is None:
            span = self.derive_span()
        if vec_length is None:
            vec_length = self.nobs
        if span >= 1:
            alpha = 2 / (span + 1)
        else:
            LOGGER.warn('Span must be >=1')
        T = vec_length
        wt_vec = {t: (1 - alpha) ** (T - t - 1) for t in range(T)}
        # make sure wt_vec = 1 for most recent obs
        self.weights = pd.DataFrame(wt_vec.values(), index=wt_vec.keys(), columns=['exp_weights'])
        #  we do not need self.weights = pd.DataFrame(np.ones((vec_length, 1)), index=range(vec_length))

    def update(self, y: np.ndarray, x: np.ndarray) -> None:
        ''' Update single obs or do bulk update '''
        x = super()._array_convert(x)
        y = super()._vec_convert(y)
        new_nobs = len(y)
        # self.y_hat_t_t. append list(np.nan * np.ones(new_nobs)))  #
        for t in range(new_nobs):  # t<new_nobs-1
            y_slice = y[t]
            x_slice = self._column_vector(x[t, :])
            self._run_update(y_slice, x_slice, save=True)
        return

    def _run_update(self, y_t: np.ndarray,
                    x_t: np.ndarray, save=True):

        x_t = self._column_vector(x_t)
        y_t = self._array2float(y_t)
        if len(x_t) != len(self.beta.state):
            print('_run_update data is of wrong dimension')
            raise FloatingPointError

        self._update_data_vec(y_t, x_t, save=save)

        # saved_handler = np.seterrcall(self.err_handler)
        # save_err = np.seterr(over='call', under='ignore', divide='call')
        y_hat_t_tminus1 = self.generate_prediction(x_t)
        prediction_error_t_tminus1 = (y_t - y_hat_t_tminus1)

        # TODO: make self.p formula = inv(X'X) wtd so stderrr
        self.p = ((1 / self.forgetting_factor) * self.p - x_t @ x_t.T)
        span = self.derive_span()
        self.exp_weights(self.nobs, span)
        # self.weights.index = endog_norm.index
        '''
        Run batch wtd regression, no ridge term
        '''

        restricted_ols_model = sm.WLS(self.y.history, self.x.history.T, self.weights)  # elastic net here too
        if self.regularization <= LAMBDA_TOL:
            restricted_fit = restricted_ols_model.fit()  # read out 'rolling' t-stats
        else:
            restricted_fit = restricted_ols_model.fit_regularized(method='elastic_net',
                                                                  alpha=self.regularization,
                                                                  L1_wt=0.0)
        self.beta.update(restricted_fit.params, save=save)

        # mse = restricted_fit.mse_total
        # df = restricted_fit.df_model  # deg of freedom
        #
        # tvalues = restricted_fit.tvalues  # regularised fitt does not have tvalues
        #
        # prediction_in_sample = pd.DataFrame(restricted_fit.fittedvalues, index=exog.index, columns=['pred_is'])
        # resids = endog - restricted_fit.fittedvalues  # restricted fit does not have resid attrib
        # resids.name = 'resids'

        y_hat_t_t = self.generate_prediction(x_t)
        super()._update_remaining_state_vars(y_hat_t_tminus1=y_hat_t_tminus1,
                                             y_hat_t_t=y_hat_t_t,
                                             prediction_error_t_tminus1=prediction_error_t_tminus1)

    def _update_data_vec(self, y_t, x_t, save=True):
        self.nobs += 1
        # print(self.x.history)
        # print(x_t)
        self.x.update(x_t, save=save)
        self.y.update(y_t, save=save)
        self.y_mean.update(y_t, save=save)
        self.y_var.update((y_t - self.y_mean.state) ** 2, save=save)


DELTA = 1.0E-16  # epsilon /delta very small number


class CR_RLS(EWRLSRidge):
    ALPHA_CONSTANT = 100  # 1000 #E3 #100 #1E5
    # 1/alpha >= normalized betas  -> beta= 0 approx
    EPSILON = 1.0e-10

    # TODO: Reparamaterize so that eta = xi(1-lambda) is a parameter!
    def __init__(self, num_features: int,
                 span: float = np.inf,
                 regularization: float = 0.0,
                 sparse_regularization: float = 0.0,
                 dynamic_regularization: bool = False,
                 scale_l2: bool = True,
                 scale_sparse: bool = True,
                 regularization_type: str = 'l0',
                 history=False,
                 beta_start=np.array([])):
        LOGGER.info('Started an EWRLS instance for {} features'.format(num_features))
        super().__init__(num_features=num_features, span=span, regularization=regularization, history=history,
                         scale_l2=scale_l2, beta_start=beta_start)
        MAX_L1NORM = 10.0  # hoping large enough
        reg_dict = {  # f, subgradient_f, rho (max f(beta) ?)
            'l1': (self.l1_norm, self.l1_subgradient, MAX_L1NORM * self.num_features),
            'alt_l1': (self.l1_norm, self.alt_l1_subgradient, MAX_L1NORM * self.num_features),
            'l0': (self.pseudo_l0_norm, self.pseudo_l0_subgradient, 1.10 * self.num_features)}
        # TODO: PASS USER DEFINED convex penalty subgradient

        # save add'l initialization parameters
        self.sparse_regularization = sparse_regularization
        self.regularization_type = regularization_type
        self.scale_sparse = scale_sparse
        self.dynamic_regularization = dynamic_regularization
        if self.dynamic_regularization:
            self.unregularized = EWRLSRidge(num_features=num_features, span=span, regularization=regularization,
                                            history=history, scale_l2=scale_l2, beta_start=beta_start)
            self.gamma_n_history = np.array([])
        else:
            self.unregularized = None

        if self.sparse_regularization == 0:
            # TODO: can we just return EWRLS object and not override anything?
            raise Exception('Non-sparse - use EWRLSRidge instead')

        (self.regularization_term,
         self.subgradient,
         self.rho) = reg_dict.get(regularization_type, (None, None, 0.0))
        if self.subgradient is None:
            raise Exception('Must choose one of the l0, l1, or alt_l1 reg funcs')

    def __repr__(self):
        return f'CR_RLS("span={self.span}", "l2-regularization={self.regularization}",' \
               f' sparse_regularization={self.sparse_regularization},' \
               f' regularization_type={self.regularization_type}. ' \
               f' dynamic_regularization={self.dynamic_regularization}' \
               f' features={self.num_features})'

    def reset(self, span=None, regularization=None, sparse_regularization=None,
              regularization_type=None):
        if self.dynamic_regularization:
            # reset history of unreg as well
            self.unregularized.reset(span=span,
                                     regularization=regularization)

        if sparse_regularization is None:
            sparse_regularization = self.sparse_regularization
        if regularization_type is None:
            regularization_type = self.regularization_type
        if span is None:
            span = self.span
        if regularization is None:
            regularization = self.regularization

        self.__init__(num_features=self.num_features,
                      span=span,
                      regularization=regularization,
                      sparse_regularization=sparse_regularization,
                      regularization_type=regularization_type,
                      dynamic_regularization=self.dynamic_regularization,
                      scale_l2=self.scale_l2,
                      scale_sparse=self.scale_sparse,
                      history=self.history,
                      beta_start=self.beta_start)

    def cross_validation(self, y_train, x_train,
                         verbose=True,
                         span_bounds=const.span_bounds,  # (1 + SPAN_TOL, 5E10),
                         ridge_reg_bounds=const.ridge_reg_bounds,  # (LAMBDA_TOL, 1E3),
                         sparse_reg_bounds=const.sparse_reg_bounds,  # (1E-2, 1E5),
                         window=np.inf):
        # save the data, not used for training yet tho
        self.y_train = y_train
        self.x_train = x_train
        self.cross_validation_verbose = verbose
        self.original_span = self.span
        self.original_regularization = self.regularization
        self.original_sparse_regularization = self.sparse_regularization

        def cv_objective(self, span=None, regularization=None,
                         sparse_regularization=None):
            # default to original span and regularization
            nobs = len(self.y_train)  # self.nobs
            # TODO: check if pass None to reset? Also in parent class
            self.reset(span=span,
                       regularization=regularization,
                       sparse_regularization=sparse_regularization)

            if (window < np.inf):
                cutoff = max(nobs - window, 0)
                self.update(self.y_train[:cutoff], self.x_train[:cutoff, :])
                sse1 = self.sse
                nobs1 = self.nobs
                self.update(self.y_train[cutoff:], self.x_train[cutoff:, :])
                sse2 = self.sse
                nobs2 = self.nobs
                windowed_mse = (sse2 - sse1) / (nobs2 - nobs1)
                self.windowed_mse = windowed_mse
            else:
                self.update(self.y_train, self.x_train)
                windowed_mse = self.mse
                self.windowed_mse = windowed_mse

            if self.cross_validation_verbose:
                LOGGER.info('Span = {}'.format(span) + ' Reg = {}'.format(regularization) +
                            ' Sparse = {}'.format(sparse_regularization) +
                            ' MSE = {}'.format(windowed_mse))
            # TODO: ever an np.array? (or just in parent class). If so, why?
            if isinstance(windowed_mse, np.ndarray):
                windowed_mse = windowed_mse[0]
            return windowed_mse

        warnings.filterwarnings('ignore', '.*objective has been.*', )
        res = gbrt_minimize(lambda x: cv_objective(self, x[0], x[1], x[2]),  # the function to minimize
                            [span_bounds,
                             ridge_reg_bounds,
                             sparse_reg_bounds],  # the bounds on each dimension of x
                            x0=[self.original_span,
                                self.original_regularization,
                                self.original_sparse_regularization],  # the starting point
                            acq_func="LCB",  # "gp_hedge",  # the acquisition function (optional)
                            acq_optimizer='lbfgs',
                            n_calls=100,  # the number of evaluations of f including at x0
                            n_initial_points=10,
                            # n_random_starts=15,  # the number of random initial points
                            # # verbose=verbose,
                            random_state=789)
        self.cv_results = res
        windowed_mse = res.fun
        # don't leave at last evaluated point, reset to optimal
        self.reset(span=res.x[0], regularization=res.x[1], sparse_regularization=res.x[2])
        self.windowed_mse = windowed_mse  # just in case since we will lose all info
        if verbose:
            LOGGER.info('Optimal CV Error at span = {}'.format(self.span) +
                        ' reg = {}'.format(self.regularization) +
                        ' sparse_reg = {}'.format(self.sparse_regularization) +
                        ' mse = {}'.format(self.windowed_mse)
                        )

    def _run_update(self, y_t: np.ndarray, x_t: np.ndarray, save=True):
        '''
        override EWRLS _run_update() using convex regularization
        (Reference: Eksioglu, and Tanc, RLS Algorithm with Convex Regularization,
        IEEE Signal Proc Letters, vol 18, no 8, Aug 2011)
        '''
        if self.dynamic_regularization:
            # from Eksiolglu-Tanc, RLS Algo with Convex Reg
            beta_0 = self.unregularized.beta.state
            beta_hat = self.beta.state
            epsilon_prime = beta_hat - beta_0
            denom = np.linalg.norm(self.p @ self.subgradient(beta_hat), 2) ** 2
            if abs(denom) > 1E-16:
                gamma_n = (2 * (np.trace(self.p) / self.num_features * (self.regularization_term(beta_hat) - self.rho)
                                + self.subgradient(beta_hat).T @ self.p @ epsilon_prime) / denom)
            else:
                gamma_n = 0.0
            gamma_n = max(gamma_n, 0)
            self.gamma_n_history = np.append(self.gamma_n_history, np.array([gamma_n]))
            self.unregularized._run_update(y_t, x_t, save=True)
        x_t = self._column_vector(x_t)
        y_t = self._array2float(y_t)
        if len(x_t) != len(self.beta.state):
            LOGGER.warn('_run_update data is of wrong dimension')
            raise FloatingPointError

        k_gain, norm_factor = self._gain(x_t)
        # generate forecast before updating beta.
        # y_hat_t_tminus1 is E[y(t) | x(t), y(1:t-1)]
        #    = beta[x[1:t-1],y[1:t-1]] * x[t]
        y_hat_t_tminus1 = self.generate_prediction(x_t)
        # y_hat_t_tminus1 = x_t beta_t_tminus1
        # note beta_t_tminus1 = beta_(t-1)_(t-1) (no transition matrix)
        prediction_error_t_tminus1 = (y_t - y_hat_t_tminus1)
        scale_p = (1 / self.forgetting_factor) if self.scale_l2 else 1
        self.p = (scale_p * self.p - k_gain * k_gain.T * norm_factor)
        #    p - p.dot(x_t) * c * x_t.T.dot(P)
        eta = (self.sparse_regularization * (1 - self.forgetting_factor)
               if self.scale_sparse else self.sparse_regularization)
        if self.dynamic_regularization:
            eta = eta * gamma_n
            # todo: can we set self.sparse_regularization = 1? what about CV?
        beta_update = (self.beta.state + k_gain * prediction_error_t_tminus1
                       - eta * (self.p @ self.subgradient(self.beta.state)))
        self.beta.update(beta_update, save=save)
        # kx1 vector beta_t_t
        y_hat_t_t = self.generate_prediction(x_t, ndarray=False)
        self._update_data_vec(y_t, x_t, save=save)
        self._update_remaining_state_vars(y_hat_t_tminus1=y_hat_t_tminus1,
                                          y_hat_t_t=y_hat_t_t,
                                          prediction_error_t_tminus1=prediction_error_t_tminus1,
                                          save=save)

    def l1_subgradient(self, x: np.array) -> np.array:
        # technically @staticmethod
        return np.sign(x)

    def alt_l1_subgradient(self, x: np.array) -> np.array:
        eps = self.EPSILON
        return np.sign(x) / (np.abs(x) + eps)

    def pseudo_l0_subgradient(self, x: np.array) -> np.array:
        alpha = self.ALPHA_CONSTANT
        # z = alpha * np.sign(x) - alpha ** 2 * x
        z = alpha * np.sign(x) * np.exp(-1 * alpha * np.abs(x))
        z[np.abs(x) > (1 / alpha)] = 0.0
        return z

    def pseudo_l0_norm(self, x: np.array) -> float:
        alpha = self.ALPHA_CONSTANT
        f = alpha * np.sum(1 - np.exp(-1 * alpha * np.abs(x)))
        return f

    def l1_norm(self, x: np.array) -> float:
        f = np.linalg.norm(x, 1)
        return f


TEST_RLS_CV = False


def main():
    '''
    Generate random data
    '''
    # bench_ols_rls1()
    model_betas = dict()
    x = np.random.randn(200, 2)
    beta = np.array([[2, -1]])
    y = x @ beta.T + np.random.randn(200, 1) * 0.05
    model = EWRLSRidge(num_features=2, span=90, regularization=30, history=True, calc_stderr=True)
    model.update(y[:90], x[:90, :])
    print(model.stderr.total_history.T)
    # print(model.beta.total_history)

    model_batch = EWRLSBatch(num_features=2, span=90, regularization=30, history=True)
    model_batch.update(y[:90], x[:90, :])

    model_betas['Batch'] = pd.DataFrame(model_batch.beta.total_history.T)
    model_betas['RLS'] = pd.DataFrame(model.beta.total_history.T)
    total_betas = pd.concat(model_betas, axis=1)
    LOGGER.info(total_betas)
    LOGGER.info(total_betas.std())

    if TEST_RLS_CV:
        model.reset()
        model.cross_validation(y[:180], x[:180, :], verbose=True)
        model.update(y[:90], x[:90, :])

    x_sparse = np.append(x, np.random.randn(200, 3), axis=1)
    model_sparse = CR_RLS(num_features=5, span=1000000, regularization=0.0001,
                          sparse_regularization=5,
                          regularization_type='l0',  # 'alt_l1',
                          dynamic_regularization=True,
                          history=True)
    model_sparse.update(y[:90], x_sparse[:90, :])
    model_sparse.reset()
    model_sparse.cross_validation(y[:180], x_sparse[:180, :], verbose=True)
    model_sparse.update(y[:180], x_sparse[:180, :])  # using 'optimal'
    LOGGER.info(pd.DataFrame(model_sparse.beta.total_history.T))


if __name__ == '__main__':
    main()

    '''


from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm#42926270
alpha=2/(span+1)
copied from pandas ewma
def get_center_of_mass(
    comass: Optional[float],
    span: Optional[float],
    halflife: Optional[float],
    alpha: Optional[float],
) -> float:
    valid_count = count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2.0
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1.0 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


def count_not_none(*args) -> int:
    """
    Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)



'''
