from typing import Callable, Tuple

import numpy as np
import pandas as pd


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
