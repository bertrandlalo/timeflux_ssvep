import mne
import numpy as np
from moabb.datasets import SSVEPExo
from numpy.lib import stride_tricks

dataset = SSVEPExo()
event_id = dataset.event_id


def extract_epochs(raw, tmin=1, tmax=5, event_id=event_id):
    # Extract epochs
    events = mne.find_events(raw, verbose=False)

    picks = mne.pick_types(raw.info, eeg=True)
    epochs_all = mne.epochs.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                                   tmax=tmax, baseline=None, verbose=False, picks=picks)
    X = epochs_all.get_data() * 1e6
    y = epochs_all.events[:, -1]
    return X, y


def epoch(a, size, interval, axis=-1):
    """ Small proof of concept of an epoching function using NumPy strides
    License: BSD-3-Clause
    Copyright: David Ojeda <david.ojeda@gmail.com>, 2018
    Create a view of `a` as (possibly overlapping) epochs.
    The intended use-case for this function is to epoch an array representing
    a multi-channels signal with shape `(n_samples, n_channels)` in order
    to create several smaller views as arrays of size `(size, n_channels)`,
    without copying the input array.
    This function uses a new stride definition in order to produce a view of
    `a` that has shape `(num_epochs, ..., size, ...)`. Dimensions other than
    the one represented by `axis` do not change.
    Parameters
    ----------
    a: array_like
        Input array
    size: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    axis: int
        Axis of the samples on `a`. For example, if `a` has a shape of
        `(num_observation, num_samples, num_channels)`, then use `axis=1`.
    Returns
    -------
    ndarray
        Epoched view of `a`. Epochs are in the first dimension.
    """
    a = np.asarray(a)
    n_samples = a.shape[axis]
    n_epochs = (n_samples - size) // interval + 1

    new_shape = list(a.shape)
    new_shape[axis] = size
    new_shape = (n_epochs,) + tuple(new_shape)

    new_strides = (a.strides[axis] * interval,) + a.strides

    return stride_tricks.as_strided(a, new_shape, new_strides)
