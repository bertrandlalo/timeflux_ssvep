"""Estimation of SSVEP covariances """
import functools

import numpy as np
from pyriemann.clustering import Potato
from pyriemann.utils.covariance import covariances
from scipy.signal import iirfilter, firwin, filtfilt
from scipy.stats import norm, chi2
from sklearn.base import BaseEstimator, TransformerMixin


def design_filter(frequencies: list, order: int, fs: float = 1.0, filter_design="fir",
                  filter_type: str = 'bandpass', **kwargs):
    """ FIR / IIR filter design given order and critical points.

    Parameters
    ----------
    order: int
        Order of the FIR filter.
    frequencies: float | list
        Cutoff frequency of filter OR a list of cutoff frequencies (that is, band edges).
     fs: scalar
        Nominal rate of the data.
    filter_design: str
        Design of filter to use ('iir' or 'fir'). Default to 'fir'.
    filter_type: str
         Type of filter to use ('bandpass', 'bandstop', 'lowpass' or 'highpass').
         Default to 'bandpass'.
    kwargs:
        Keywords aruments to pass to the filter design functions:
            - when filter_type is 'fir', see additional keyword args such as rp, rs, ..., at
            scipy.signal.iirfilter:<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.iirfilter.html>
            - when filter_type is 'iir', see additional keyword args width, window, pass_zero and scale
            at scipy.signal.firwin:<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.firwin.html>

    Returns
    -------
    b, a : ndarray, ndarray
            Numerator (`b`) and denominator (`a`) polynomials of the filter.
    """
    if filter_type not in ['bandpass', 'bandstop', 'lowpass', 'highpass']:
        raise ValueError('type should be bandpass, bandstop, highpass or lowpass')

    if filter_design not in ['fir', 'iir']:
        raise ValueError('design should be fir or iir')

    # define the function to apply to the selected columns
    if filter_design == 'fir':
        if filter_type == 'bandpass':
            b = firwin(numtaps=order + 1, cutoff=frequencies, fs=fs, pass_zero=False, **kwargs)
        elif filter_type == 'bandstop':
            b = firwin(numtaps=order + 1, cutoff=frequencies, fs=fs, pass_zero=True, **kwargs)
        elif filter_type == 'highpass':
            b = firwin(numtaps=order + 1, cutoff=frequencies, fs=fs, pass_zero=False, **kwargs)
        elif filter_type == 'lowpass':
            b = firwin(numtaps=order + 1, cutoff=frequencies, fs=fs, pass_zero=True, **kwargs)
        a = 1
        filter_args = (b, a)
    else:  # filter_design == 'iir':
        kwargs.setdefault('output', 'ba')
        filter_args = iirfilter(order, Wn=frequencies, btype=filter_type, fs=fs, **kwargs)
    return filter_args


class SSVEPCovariance(BaseEstimator, TransformerMixin):
    """Estimate special form covariance matrix for SSVEP.

    Perform a simple covariance matrix estimation for each given trial.

    Parameters
    ----------
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.

    """

    def __init__(self, rate: float = 256, flickering_frequencies: tuple = (13, 17, 21),
                 estimator: str = 'scm', **kwargs):
        """Init."""
        self._estimator = estimator
        self._design_filters(rate, flickering_frequencies, **kwargs)

    def fit(self, X: np.array, y: np.array = None) -> object:
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : SSVEPCovariance instance
            The SSVEPCovariance instance.
        """
        return self

    def transform(self, X: np.array) -> np.array:
        """Estimate covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels * n_frequencies, n_channels * n_frequencies)
            SSVEP covariance matrices for each trials.
        """
        filtered = []
        for filter in self.filters:
            filtered_X = filter(X, axis=-1)
            filtered.append(filtered_X)
        mega_X = np.concatenate(filtered, axis=1)
        out = covariances(mega_X, estimator=self._estimator)
        return out

    def _design_filters(self, fs: float, flickering_frequencies: tuple,
                        bandwidth: float = 4, order: int = 2,
                        filter_design: str = 'iir', padlen: int = None):
        self.filters = []
        for flickering_frequency in flickering_frequencies:
            frequencies = [flickering_frequency - bandwidth / 2,
                           flickering_frequency + bandwidth / 2]  # TODO/Question : shall we consider harmonics as well ?
            filter_args = design_filter(frequencies=frequencies, order=order, fs=fs, filter_design=filter_design)
            filter_func = functools.partial(filtfilt, *filter_args, padlen=padlen)
            self.filters.append(filter_func)


class SSVEPPotato(BaseEstimator, TransformerMixin):
    """Estimate potato field per class
    """

    def __init__(self, classes: tuple = None, potato_threshold: float = .1, field_threshold: float = .1,
                 n_iter_max: int = 100, pos_label=1, neg_label=0):
        """Init."""
        self.classes = classes
        self.potato_threshold = potato_threshold
        self.field_threshold = field_threshold
        self.n_iter_max = n_iter_max
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, X: np.array, y: np.array) -> object:
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : SSVEPPotato instance
            The SSVEPPotato instance.
        """
        self.classes = self.classes or set(y)
        self._n_potatoes = len(self.classes)
        self._potatoes = []
        for _class in self.classes:
            X_class = X[y == _class, :, :]
            potato = Potato(n_iter_max=self.n_iter_max, threshold=self.potato_threshold).fit(X_class)
            self._potatoes.append(potato)

        return self

    def transform(self, X: np.array) -> np.array:
        """return the SQI, combining the output z-scores of the potatoes

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        p : ndarray, shape (n_epochs, 1)
         p ∈ [0, 1] is the SQI, combining the output z-scores of the potatoes.
        """

        # See equations 15, 16, 17 of paper:<https://hal.archives-ouvertes.fr/hal-02015909/document>
        z_to_p = lambda z_score: 1 - norm.cdf(z_score)
        q_to_p = lambda p_value: 1 - chi2.cdf(p_value, df=self._n_potatoes)
        n_trials, n_channels, n_samples = X.shape

        q = np.zeros(n_trials)
        for potato in self._potatoes:
            z_score = potato.transform(X)
            q += np.log(np.apply_along_axis(z_to_p, axis=0, arr=z_score))
        p = np.apply_along_axis(q_to_p, axis=0, arr=2 * q)
        return p

    def predict(self, X):
        """predict artifacts from data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of bool, shape (n_epochs, 1)
            the artifact detection. True if the trial is clean, and False if
            the trial contain an artifact.
        """
        p = self.transform(X)
        pred = p < self.field_threshold
        out = np.zeros_like(p) + self.neg_label
        out[pred] = self.pos_label
        return out
