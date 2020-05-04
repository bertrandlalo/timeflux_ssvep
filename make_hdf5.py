import json
import os

import mne
import numpy as np
import pandas as pd
from moabb.datasets import SSVEPExo


def raw_to_frames(raw: mne.io.fiff.raw.Raw, stim_to_event: dict) -> (pd.DataFrame, pd.DataFrame):
    """Convert mne raw to events and eeg DataFrames

    Parameters
    ----------
    raw: raw object with channels of type 'eeg' en 'stim'
    stim_to_event: dict giving correspondence between stim values
     in raw and events row in DataFrame.

    Returns
    -------
    df_eeg: DataFrame with columns are EEG sensor and rows are samples. Index are integers
    df_events: DataFrame with columns are keys given by `stim_to_event`

    Examples
    --------
    >>> eeg = np.random.rand(10, 3)
    >>> stim = np.array([0, 0, 0, 1, 0, 0, 2, 0, 0, 0.])
    >>> data = np.hstack([eeg, stim.reshape(-1, 1)])
    >>> info = mne.create_info(ch_names=['eeg1', 'eeg2', 'eeg3', 'stim'], sfreq=1, ch_types=['eeg'] * 3 + ['stim'])
    >>> raw = mne.io.RawArray(data.T, info)
    >>> stim_to_event = {1: {'label': 'stim_1'}, 2: {'label': 'stim_2'},  'default': {'label': np.NaN}}
    >>> df_eeg, df_events = raw_to_frames(raw, stim_to_event)
    >>> df_eeg
        channel           eeg1           eeg2           eeg3
        time
        0        503924.786313  512910.408283  685353.585551
        1000     969342.747335  289300.959243  673613.527348
        2000     627263.697504  601651.504107  492553.490725
        3000     721392.710208  468848.235309  468505.768179
        4000     451762.311697  239765.139012  764511.177061
        5000     479186.259274  694454.767679  784551.272967
        6000     307081.473812  923345.961927   16373.151217
        7000      30046.972908  658291.675710  724231.828692
        8000     122259.361421  180202.733713  916851.577241
        9000      10142.254930  904170.923145  351227.238482
    >>> df_events
               label
        time
        0        NaN
        1000     NaN
        2000     NaN
        3000  stim_1
        4000     NaN
        5000     NaN
        6000  stim_2
        7000     NaN
        8000     NaN
        9000     NaN
    """

    df_eeg = raw.to_data_frame(picks=mne.pick_types(raw.info, eeg=True, stim=False))
    raw_stim = mne.find_events(raw)
    events_stim = pd.DataFrame(list(map(lambda stim: stim_to_event.get(stim), list(raw_stim[:, 2]))))
    df_events = pd.DataFrame(index=df_eeg.index, columns=events_stim.columns)
    df_events.iloc[raw_stim[:, 0], :] = events_stim.values
    return df_eeg, df_events


def runs_to_frames(session_data: dict,
                   runs: tuple,
                   stim_to_event: dict) -> (pd.DataFrame, pd.DataFrame):
    """ Concatenate multiple runs

    Parameters
    ----------
    session_data: dict with keys are run names and values are Raw objects
    runs: tuple with run names to concatenate
    stim_to_event: correspondence between stim and events. See `raw_to_frames`

    Returns
    -------
    df_eeg: DataFrame with columns are EEG sensor and rows are samples. Index are integers
    df_events: DataFrame with columns are keys given by `stim_to_event`
    """
    df_eeg = []
    df_events = []
    for run in runs:
        raw = session_data.get(run)
        if raw is None:
            raise ValueError(f'Could not find run {run} in session data.')
        df_run_eeg, df_run_events = raw_to_frames(raw, stim_to_event)
        df_eeg.append((df_run_eeg))
        df_events.append((df_run_events))
    return pd.concat(df_eeg).reset_index(drop=True), pd.concat(df_events).reset_index(drop=True)


def make_subject_hdf5(session_data: dict, output: str,
                      stim_to_event: dict,
                      train_runs: tuple, test_runs: tuple = None,
                      onset: str = '2020-01-01',
                      sampling_rate: float = 256,
                      force: bool = False):
    """Make timeflux-replayable HFD5 file from subject MOABB data

    Parameters
    ----------
    session_data: dict with keys are run names and values are Raw objects
    output: output directory with file name
    stim_to_event: correspondence between stim and events. See `raw_to_frames`
    train_runs: tuple with name of runs that should be considered as 'calibration phase'
    test_runs: tuple with name of runs that should be considered as 'test phase'.
        If None, all available runs outside  list `train_run`.
    onset: onset to infer datetime index
    sampling_rate: Sampling rate of eeg
    force: whether or not to check if HDF5 output already exists.
        If True, re-generate anyway. If False, return when exists.

    """
    # if not force, then just check the hdf5 file exists
    if not force and os.path.exists(output):
        return

    test_runs = test_runs or tuple(set(session_data.keys() - set(train_runs)))

    # compute eeg and events as DataFrame by concatenating given runs from a session
    df_eeg_train, df_events_train = runs_to_frames(session_data, train_runs, stim_to_event=stim_to_event)
    df_eeg_test, df_events_test = runs_to_frames(session_data, test_runs, stim_to_event=stim_to_event)

    # add train sequence events
    df_events_train.loc[df_events_train.index[df_events_train.dropna().index[0] - 1], 'label'] = 'train_starts'
    df_events_train.loc[df_events_train.index[-1], 'label'] = 'train_stops'

    # concatenate train data and test data (to simulate a real experiment)
    df_eeg = pd.concat([df_eeg_train, df_eeg_test])
    df_events = pd.concat([df_events_train, df_events_test])

    # convert index to uniform datetime given sampling rate
    period_ns, fract = divmod(1e9, sampling_rate)
    times = pd.date_range(onset, periods=len(df_events), freq=f'{period_ns}N')
    df_events.index = times
    df_eeg.index = times

    # drop NaN from events to avoid mixed types in hdf5
    df_events.dropna(inplace=True, how='all')

    # save to hdf5 stores
    df_eeg.to_hdf(output, '/eeg', format='table')
    df_events.to_hdf(output, '/events', format='table')


def make_subjects_hdf5(**kwargs):
    dataset = SSVEPExo()
    data = dataset.get_data()

    for subject in data:
        session_data = data[subject]['session_0']
        make_subject_hdf5(session_data, output=f'data/{subject}.hdf5', **kwargs)


if __name__ == "__main__":
    onset_label = "flickering_starts"
    data_key = "target"
    stim_to_event = {
        1: {'label': onset_label, 'data': json.dumps({data_key: "13Hz"})},
        2: {'label': onset_label, 'data': json.dumps({data_key: "17Hz"})},
        3: {'label': onset_label, 'data': json.dumps({data_key: "21Hz"})},
        4: {'label': onset_label, 'data': json.dumps({data_key: "rest"})},
        'default': {'label': np.NaN, 'data': np.NaN},
    }

    # execute only if run as a script
    make_subjects_hdf5(force=True, train_runs=('run_0',), test_runs=None, stim_to_event=stim_to_event)
