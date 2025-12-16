
import scipy.stats as stats
from pathlib import Path
from itertools import product
import pickle
from collections.abc import Iterable
import numpy as np
from mtrf.stats import (
    _crossval,
    _progressbar,
    _check_k,
    neg_mse,
    pearsonr,
)
from sklearn.metrics import r2_score
from mtrf.matrices import (
    covariance_matrices,
    banded_regularization,
    regularization_matrix,
    lag_matrix,
    truncate,
    _check_data,
    _get_xy,
)

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    import mne
except ModuleNotFoundError:
    mne = None





def predict_time_constrained_trial(
        self,
        stimulus=None,
        response=None,
        lag=None,
        average=True,
        TC=True,
        onsetVecDim=None,
        onsetVecWinSt=None,
        onsetVecWinEnd=None,
        external_dim=False,
        external_ind=None,
        skip_first=False
    ):
        """
        Predict response from stimulus (or vice versa) using the trained model.

        The matrix of TRF weights is multiplied with the time-lagged input to predict
        the output. If the actual output is provided, this method will estimate the
        correlation and mean squared error of the predicted and actual output. 
        The function additionally allows to define relevant time points relative to onsets in 
        a given feature. 

        Parameters
        ----------
        stimulus: None or list or numpy.ndarray
            Either a 2-D samples-by-features array, if the data contains only one trial
            or a list of such arrays of it contains multiple trials. The second
            dimension can be omitted if there only is a single stimulus feature
            (e.g. envelope). When using a forward model, this must be specified.
        response: None or list or numpy.ndarray
            Either a 2-D samples-by-channels array, if the data contains only one
            trial or a list of such arrays of it contains multiple trials. Must be
            provided when using a backward model.
        lag: None or in or list
            If not None (default), only use the specified lags for prediction.
            The provided values index the elements in self.times.
        average: bool or list or numpy.ndarray
            If True (default), average metric across all predicted features (e.g. channels
            in the case of forward modelling). If `average` is an array of indices only
            average those features. If `False`, return each predicted feature's metric.
        TC: bool 
            If True, the function will return a time-constrained prediction metric. 
        onsetVecDim: 
            The feature index of the feature to use for time constrained prediction. 
        onsetVecWinSt: 
            The time (in ms) of the start of the window of interest. 
        onsetVecWinEnd: 
            Time end of the window of interest. 
        external_dim: bool
            If True, the function accepts an additional array of the same length to use 
            for the time-constrained prediction metric. 
        external_ind: 
            Array to use as a reference to define the relevant time windows. 
        skip_first: bool
            If True, the first index in each stimulus is skipped for the time-constrained 
            prediction metric. 

        Returns
        -------
        prediction: numpy.ndarray
            Predicted stimulus or response
        metric: float or numpy.ndarray
            If both stimulus and response are provided, metric is computed by the
            metric function defined in the attribute `TRF.metric`.
            If average is False, an array containing the metric for each feature
            is returned.
        tc_metric: array 
            Time-constrained prediction metric
        prediction_resid: 
            residualized prediction (actual response - predicted response)
        rsq: 
            R squared values
        """
        # check that inputs are valid
        if self.weights is None:
            raise ValueError("Can't make predictions with an untrained model!")
        if self.direction == 1 and stimulus is None:
            raise ValueError("Need stimulus to predict with a forward model!")
        elif self.direction == -1 and response is None:
            raise ValueError("Need response to predict with a backward model!")
        else:
            stimulus, response, n_trials = _check_data(stimulus, response)
        if stimulus is None:
            stimulus = [None for _ in range(n_trials)]
        if response is None:
            response = [None for _ in range(n_trials)]

        x, y = _get_xy(stimulus, response, direction=self.direction)
        prediction = [np.zeros((x_i.shape[0], self.weights.shape[-1])) for x_i in x]
        prediction_resid = [np.zeros((x_i.shape[0], self.weights.shape[-1])) for x_i in x]

    
        metric = []
        tc_metric = []
        rsq = []
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            
            lags = list(
                range(
                    int(np.floor(self.times[0] * self.fs)),
                    int(np.ceil(self.times[-1] * self.fs)) + 1,
                )
            )
            w = self.weights.copy()
            if lag is not None:  # select lag and corresponding weights
                if not isinstance(lag, Iterable):
                    lag = [lag]
                lags = list(np.array(lags)[lag])
                w = w[:, lag, :]
            


            w = np.concatenate(
                [
                    self.bias,
                    w.reshape(
                        x_i.shape[-1] * len(lags), self.weights.shape[-1], order="F"
                    ),
                ]
            ) * (1 / self.fs)
            x_lag = lag_matrix(x_i, lags, self.zeropad)
            y_pred = x_lag @ w
            



            ###If using TC
            if TC == True:

                st = round(onsetVecWinSt/1000*self.fs)
                fin = round(onsetVecWinEnd/1000*self.fs)
                interval = range(st,fin)
    
                if external_dim == True:
                    ind1 = np.nonzero(external_ind[i][:,onsetVecDim])
                else:
                    ind1 = np.nonzero(x_i[:, onsetVecDim])
                ind1 = ind1[0]

                if len(ind1) > 0:
      
                    if skip_first == True:
                        ind1 = ind1[1:]
                    idx = []
                    for iin in interval:
                        for ix in ind1:
                            
                            idx.append(ix+iin)
                    idx = np.unique(idx)

                    if any(idx > len(x_i[:, onsetVecDim])-1):
                    
                        ind_f = np.where(idx > len(x_i[:, onsetVecDim])-1, True, False)
                        
                        idx[ind_f] = len(x_i[:, onsetVecDim])-1
                        idx = np.unique(idx)
                    
                


            if y_i is not None:
                if self.zeropad is False:
                    y_i = truncate(y_i, lags[0], lags[-1])
                
                ##If using TC
                if TC == True and len(ind1) > 0:
                    tc_metric.append(self.metric(y_i[idx,:], y_pred[idx,:]))
                elif TC == True and len(ind1) == 0:
                    tc_metric.append(np.full(shape=(61,), fill_value=np.nan))

                metric.append(self.metric(y_i, y_pred))
                rsq.append(r2_score(y_i, y_pred, multioutput='uniform_average'))

                prediction[i][:] = y_pred
                prediction_resid[i][:] = y_i - y_pred


        if y[0] is not None:

            if average is not False:
                metric = np.mean(metric, axis = 1)
                tc_metric = np.mean(tc_metric, axis = 1)
                rsq = np.mean(rsq)

            return prediction, metric,tc_metric, prediction_resid, rsq
        else:
            return prediction,tc_metric


