import numpy as np
import pandas as pd

from utilities.settings import LOGGER
from.ewrls import EWRLSRidge


class FeatureNormalizer(object):
    # TODO : create Featurenormalizer = EWRLSRidge for each component  y = mu + eps
    def __init__(self, min_sample_length=1, span=np.inf, expanding=True):
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

        self.expanding = expanding
        self.span = span
        min_sample_length = max(min_sample_length, 1)
        self.min_sample_length = min_sample_length

    def initialize(self, exog):
        if len(exog.shape) == 1:
            self.dimension = 1
            self.col_names = [exog.name]
            exog = pd.DataFrame(exog, columns=self.col_names)
        else:
            self.dimension = exog.shape[1]
            self.col_names = list(exog.columns)

        # TODO: Check if constant, and eliminate then reinstroduce!

        self.series_normalizers = {col: EWRLSRidge(num_features=1, span=self.span, regularization=0) for col
                                   in
                                   self.col_names}
        self.normalized_features = dict()
        exog_const = np.ones((len(exog), 1))
        for col in self.col_names:
            self.series_normalizers[col].update(exog[col], exog_const)
            means = pd.Series(self.series_normalizers[col].y_hat_t_t, index=exog[col].index)
            resid_ser = pd.Series([x[0] for x in self.series_normalizers[col].prediction_error_t_tminus1],
                                  index=exog[col].index)
            resid_mean = resid_ser.ewm(span=self.span).mean()
            resid_var = resid_ser.ewm(span=self.span).var()
            expwtdstd = (resid_var + resid_mean.pow(2)).pow(0.5).fillna(1)
            expwtdstd.index = exog[col].index
            # stdevs = pd.Series(self.series_normalizers[col].y_var, index=exog[col].index).pow(0.5)
            self.normalized_features[col] = (exog[col] - means).div(expwtdstd)
            # self.normalized_feature[col] = self.series_normalizers[col].y_hat_t_t
        self.in_sample_normalized = pd.concat(self.normalized_features, axis=1).fillna(method='bfill')

        # ASSERT
        # resid_ser = pd.Series([x[0] for x in self.series_normalizers[col].prediction_error_t_tminus1])
        # resid_mean = resid_ser.ewm(span=self.span).mean()
        # resid_var = resid_ser.ewm(span=self.span).var()
        # expwtsd = (resid_var + resid_mean.pow(2)).pow(0.5)
        # ASSERT expwtsd =pd.Series([x[0] for x in self.series_normalizers[col].in_sample_std]]
        # ser1 = pd.Series([x[0] for x in self.series_normalizers[col].prediction_error_t_tminus1]).ewm(span=self.span).std()
        #   ==
        # self.nobs_window = exog.shape[0]
        # if type(exog) == pd.Series:
        #     self.type = pd.Series
        # else:
        #     self.type = pd.DataFrame
        # if (self.type == pd.DataFrame) and ('const' in exog.columns):
        #     exog_non_const = exog.drop(columns=['const'])
        #     self.const = True
        # else:  # series can't remove constant
        #     exog_non_const = exog
        #     self.const = False

        # if self.expanding:
        #     exog_stdev = exog_non_const.expanding(min_periods=self.min_sample_length).std()
        #     exog_mean = exog_non_const.expanding(min_periods=self.min_sample_length).mean()
        #     if self.type == pd.Series:
        #         exog_mean.iloc[:self.min_sample_length] = np.nan
        #         exog_stdev.iloc[:self.min_sample_length] = np.nan
        #     else:
        #         exog_mean.iloc[:self.min_sample_length, :] = np.nan
        #         exog_stdev.iloc[:self.min_sample_length, :] = np.nan
        #     exog_mean = exog_mean.fillna(method='bfill')
        #     exog_stdev = exog_stdev.fillna(method='bfill')
        #     # no NANs!
        #     # set to 'in-sample/' during first period
        #     # should nan out first [1:min_periods] on BOTH mean and stdev and then bfill
        # else:  # normal, test period and others
        #     if self.min_sample_length < np.inf:
        #         if self.type == pd.DataFrame:
        #             exog_stdev = exog_non_const.iloc[:self.min_sample_length, :].std()
        #             exog_mean = exog_non_const.iloc[:self.min_sample_length, :].mean()
        #         else:  # series
        #             exog_stdev = exog_non_const.iloc[:self.min_sample_length].std()
        #             exog_mean = exog_non_const.iloc[:self.min_sample_length].mean()
        #     else:  # == np.inf
        #         exog_stdev = exog_non_const.std()
        #         exog_mean = exog_non_const.mean()
        # exog_norm = (exog_non_const - exog_mean).div(exog_stdev)
        # if self.const:
        #     exog_norm = pd.concat([exog['const'], exog_norm], axis=1)  # put the constant back in
        # if self.type == pd.Series:
        #     self.mean = exog_mean.iloc[-1]
        #     self.stdev = exog_stdev.iloc[-1]
        # else:
        #     self.mean = exog_mean.iloc[-1, :]
        #     self.stdev = exog_stdev.iloc[-1, :]
        #
        # self.nobs_window = len(exog_norm)
        # return exog_norm  # return pd.DataFrame or pd.Series

    def update(self, exog_obs):
        if self.const:
            exog_non_const = exog_obs.drop(columns=['const'])  # index?
        else:
            exog_non_const = exog_obs

        self.mean = (self.mean * self.nobs + exog_non_const) / (self.nobs + 1)
        var = self.stdev ** (2)
        var = (var * self.nobs + (exog_non_const - self.mean) ** (2)) / (self.nobs + 1)
        self.stdev = var ** 0.5
        exog_norm = (exog_non_const - self.mean) / (self.stdev)
        if self.const:
            exog_norm = pd.concat([exog_non_const['const'], exog_norm], axis=1)  # put the constant back in
        return exog_norm  # return an observation


def vol_scale_forecast(unscaled_predictions, target_response,
                       out_of_sample_prediction, forgetting_factor, spans,
                       vol_scale_cap):
    if forgetting_factor < 1:
        try:
            forecast_vol = unscaled_predictions.ewm(alpha=1 - forgetting_factor) \
                .std().fillna(1).iloc[-1]
        except:
            # print(unscaled_predictions.head())
            # print(type(unscaled_predictions))
            # print(unscaled_predictions.shape)
            raise FloatingPointError('Unable to find vol scale')
    else:
        forecast_vol = unscaled_predictions.std()
    vol_scaled_forecast = out_of_sample_prediction * min(
        (target_response.ewm(span=spans).std().iloc[-1, 0] / forecast_vol), vol_scale_cap)
    return vol_scaled_forecast


def vol_scale_series(unscaled_predictions, target_response, forgetting_factor, spans,
                     vol_scale_cap):
    length = unscaled_predictions.shape[0]
    if forgetting_factor < 1:
        try:
            forecast_vol = unscaled_predictions.ewm(alpha=1 - forgetting_factor) \
                .std().fillna(1)
        except:
            raise FloatingPointError('Unable to find vol scale')
    else:
        forecast_vol = unscaled_predictions.std() * pd.Series(np.ones((length,)),
                                                              index=unscaled_predictions.index)
    forecast_vol.name = 'norm_unscaled_forecast_vol'
    target_vol = target_response.ewm(span=spans).std()
    same_len_vol = pd.concat([target_vol, forecast_vol], axis=1).ffill()
    same_len_vol.columns = ['target_vol', 'forecast_vol']

    capped_vol_ratio = (same_len_vol['target_vol'] / same_len_vol['forecast_vol'])
    capped_vol_ratio = capped_vol_ratio.map(lambda x: min(x, vol_scale_cap))

    capped_vol_ratio[:5] = 1
    vol_scaled_forecast = unscaled_predictions * capped_vol_ratio

    return vol_scaled_forecast


def exp_weights(vec_length, span):
    if span >= 1:
        alpha = 2 / (span + 1)
    else:
        LOGGER.error('Span must be >=1')



    T = vec_length
    wt_vec = {t: (1 - alpha) ** (T - t - 1) for t in range(T)}
    # make sure wt_vec = 1 for most recent obs
    weights = pd.DataFrame(wt_vec.values(), index=wt_vec.keys(), columns=['exp_weights'])

    # inv_weights = pd.DataFrame(np.zeros((vec_length, 1)))
    # inv_weights.iloc[0] = 1
    # inv_weights = inv_weights.ewm(span=span).mean()
    # # inv_weights = inv_weights / inv_weights.sum()  # normalized EWM with fixed width
    # # now reverse order so intead of [l, l**2, l**3, .... l**T] it becomes [l**T, l**(T-1),.... l**2, l]
    # weights =pd.DataFrame(inv_weights.values[::-1], index=inv_weights.index, columns = ['exp_weights'])
    # weights = pd.DataFrame([1/x for x in exponential_weights.values], columns =['exp_wts'])
    # the help file is damn confusing! Why should you *ever* transform by 1/w.
    # The .py file is much easier to understand!
    return weights
